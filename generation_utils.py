import os
import copy
import re
import torch
import torch.nn.functional as F
import random
import numpy as np
import math


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        import transformers
        transformers.set_seed(seed)
    except Exception:
        pass


def apply_sampling_filter(logits, top_k=0, top_p=1.0, min_p=0.0):
    if top_k > 0:
        top_k_values, _ = torch.topk(logits, top_k, dim=-1)
        min_top_k = top_k_values[:, -1].unsqueeze(-1)
        logits = torch.where(logits < min_top_k, float('-inf'), logits)
    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = 0
        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits = logits.masked_fill(indices_to_remove, float('-inf'))
    if min_p > 0:
        probs = F.softmax(logits, dim=-1)
        logits = torch.where(probs < min_p, float('-inf'), logits)
    return logits


def _select_tensor_batch(value, indices):
    if value is None or not isinstance(value, torch.Tensor) or value.ndim == 0:
        return value
    return value.index_select(0, indices.to(value.device))


def _batch_select_cache_layer(layer, indices):
    has_linear_states = hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states")
    if has_linear_states and hasattr(layer, "reorder_cache"):
        layer.reorder_cache(indices)
    elif hasattr(layer, "batch_select_indices"):
        layer.batch_select_indices(indices)
    elif hasattr(layer, "reorder_cache"):
        layer.reorder_cache(indices)
    else:
        for attr in ("keys", "values", "conv_states", "recurrent_states"):
            value = getattr(layer, attr, None)
            if isinstance(value, torch.Tensor):
                setattr(layer, attr, _select_tensor_batch(value, indices))

    if hasattr(layer, "max_batch_size"):
        try:
            layer.max_batch_size = int(indices.numel())
        except Exception:
            pass


def _needs_layerwise_batch_select(past_key_values):
    layers = getattr(past_key_values, "layers", None)
    if layers is None:
        return False
    for layer in layers:
        has_linear_states = hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states")
        if has_linear_states or not hasattr(layer, "batch_select_indices"):
            return True
    return False


def batch_select_hybrid_cache(past_key_values, indices):
    if past_key_values is None:
        return past_key_values

    if hasattr(past_key_values, "batch_select_indices") and not _needs_layerwise_batch_select(past_key_values):
        past_key_values.batch_select_indices(indices)
        return past_key_values

    if hasattr(past_key_values, "layers"):
        for layer in past_key_values.layers:
            _batch_select_cache_layer(layer, indices)
        return past_key_values

    if hasattr(past_key_values, "batch_select_indices"):
        past_key_values.batch_select_indices(indices)
        return past_key_values

    if isinstance(past_key_values, tuple):
        selected_layers = []
        for layer in past_key_values:
            if isinstance(layer, tuple):
                selected_layers.append(tuple(_select_tensor_batch(v, indices) for v in layer))
            else:
                selected_layers.append(_select_tensor_batch(layer, indices))
        return tuple(selected_layers)
    return past_key_values


def _cache_layers(past_key_values):
    return getattr(past_key_values, "layers", None)


def _layer_has_linear_states(layer):
    return hasattr(layer, "conv_states") or hasattr(layer, "recurrent_states")


def cache_has_linear_layers(past_key_values):
    layers = _cache_layers(past_key_values)
    if layers is None:
        return False
    return any(_layer_has_linear_states(layer) for layer in layers)


def clone_single_cache(past_key_values, batch_idx, device):
    if past_key_values is None:
        return None
    single_cache = copy.deepcopy(past_key_values)
    keep_idx = torch.tensor([batch_idx], dtype=torch.long, device=device)
    return batch_select_hybrid_cache(single_cache, keep_idx)


def crop_cache(cache, max_length):
    if cache is None:
        return cache
    if hasattr(cache, "crop"):
        cache.crop(max_length)
    return cache


def snapshot_linear_cache_states(past_key_values, batch_idx, device):
    layers = _cache_layers(past_key_values)
    if layers is None:
        return None
    keep_idx = torch.tensor([batch_idx], dtype=torch.long, device=device)
    snapshot = []
    found = False
    for layer in layers:
        layer_snapshot = {}
        for attr in ("conv_states", "recurrent_states"):
            value = getattr(layer, attr, None)
            if isinstance(value, torch.Tensor):
                layer_snapshot[attr] = _select_tensor_batch(value, keep_idx).detach().clone()
                found = True
        snapshot.append(layer_snapshot)
    return snapshot if found else None


def restore_linear_cache_states(past_key_values, snapshot):
    if past_key_values is None or snapshot is None:
        return past_key_values
    layers = _cache_layers(past_key_values)
    if layers is None:
        return past_key_values
    for layer, layer_snapshot in zip(layers, snapshot):
        for attr, value in layer_snapshot.items():
            setattr(layer, attr, value.detach().clone())
        if layer_snapshot and hasattr(layer, "max_batch_size"):
            try:
                layer.max_batch_size = 1
            except Exception:
                pass
    return past_key_values


def restore_linear_cache_states_for_batch(past_key_values, snapshot, batch_idx):
    if past_key_values is None or snapshot is None:
        return False
    layers = _cache_layers(past_key_values)
    if layers is None:
        return False

    restored = False
    for layer, layer_snapshot in zip(layers, snapshot):
        for attr, snapshot_value in layer_snapshot.items():
            value = getattr(layer, attr, None)
            if not isinstance(value, torch.Tensor):
                continue
            if batch_idx >= value.size(0):
                continue

            snapshot_value = snapshot_value.detach().to(device=value.device, dtype=value.dtype)
            if snapshot_value.size(0) != 1:
                continue
            if value[batch_idx : batch_idx + 1].shape != snapshot_value.shape:
                continue
            with torch.no_grad():
                value[batch_idx : batch_idx + 1].copy_(snapshot_value)
            restored = True
    return restored


def generate_cot(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")

    # ---- **gen_kwargs ----
    temperature = kwargs.get("temperature", 1.0)
    top_p = kwargs.get("top_p", 1.0)
    top_k = kwargs.get("top_k", 0)
    min_p = kwargs.get("min_p", 0)
    max_new_tokens = kwargs.get("max_new_tokens", 128)
    do_sample = kwargs.get("do_sample", True)
    
    batch_size = input_ids.shape[0]
    device = input_ids.device
    embedding_layer = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight

    all_generated = [input_ids[i].clone().tolist() for i in range(batch_size)]
    unfinished_idx = list(range(batch_size))

    generated = input_ids.clone()
    attn_mask = attention_mask.clone()
    past_key_values = None
        
    for step in range(max_new_tokens):
        cur_batch = generated.shape[0]
        if cur_batch == 0:
            break

        if past_key_values is None:
            model_inputs = {"input_ids": generated, "attention_mask": attn_mask}
        else:
            # model_inputs = {"input_ids": next_tokens.unsqueeze(1), "past_key_values": past_key_values}
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attn_mask.dtype, device=device) ###
            attn_mask = torch.cat([attn_mask, attention_mask_new], dim=1) ###
            model_inputs = {"input_ids": next_tokens.unsqueeze(1), "past_key_values": past_key_values, "attention_mask": attn_mask} ###

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values

        next_token_logits = outputs.logits[:, -1, :]  # [cur_batch, vocab]
        logits = next_token_logits / temperature
        logits = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)

        probs = F.softmax(logits, dim=-1)
        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)

        for bi, orig in enumerate(unfinished_idx):
            all_generated[orig].append(next_tokens[bi].item())

        if tokenizer.eos_token_id is not None:
            cur_finished = (next_tokens == tokenizer.eos_token_id)
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]

        if len(unfinished_idx) == 0:
            break
        generated = generated[keep_idx]
        next_tokens = next_tokens[keep_idx]
        attention_mask = attention_mask[keep_idx]
        attn_mask = attn_mask[keep_idx] ###
        keep_idx_tensor = keep_idx if isinstance(keep_idx, torch.Tensor) else torch.tensor(keep_idx, dtype=torch.long, device=generated.device)
        past_key_values = batch_select_hybrid_cache(past_key_values, keep_idx_tensor)

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), tokenizer.pad_token_id or 0, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out


# For pure transformer-based models such as Qwen3 model families
def generate_copt(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")

    # ---- **gen_kwargs ----
    temperature = kwargs.get("temperature", 1.0)
    top_p = kwargs.get("top_p", 1.0)
    top_k = kwargs.get("top_k", 0)
    min_p = kwargs.get("min_p", 0)
    max_new_tokens = kwargs.get("max_new_tokens", 128)
    do_sample = kwargs.get("do_sample", True)

    batch_size = input_ids.shape[0]
    device = input_ids.device
    embedding_layer = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight

    end_of_thinking_text = kwargs.get("end_of_thinking_text", "</think>")
    end_of_thinking_ids = tokenizer.encode(end_of_thinking_text, add_special_tokens=False)
    end_of_thinking = torch.tensor(end_of_thinking_ids, dtype=input_ids.dtype, device=device)

    rebuilt_sequences = []
    for i in range(batch_size):
        valid_prompt = input_ids[i][attention_mask[i].bool()]
        rebuilt_sequences.append(torch.cat([valid_prompt, end_of_thinking], dim=0))

    max_prompt_len = max(seq.size(0) for seq in rebuilt_sequences)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    generated = torch.full((batch_size, max_prompt_len), pad_token_id, dtype=input_ids.dtype, device=device)
    attention_mask = torch.zeros((batch_size, max_prompt_len), dtype=attention_mask.dtype, device=device)
    for i, seq in enumerate(rebuilt_sequences):
        seq_len = seq.size(0)
        generated[i, -seq_len:] = seq
        attention_mask[i, -seq_len:] = 1
    all_generated = [generated[i].clone().tolist() for i in range(batch_size)]

    tau_a = kwargs.get("tau_a", 0)
    tau_r = kwargs.get("tau_r", 0)
    draft_max_new_tokens = kwargs.get("draft_max_new_tokens", 1024)
    restart_think_text = kwargs.get("restart_think_text", "<think>")
    restart_think_ids = tokenizer.encode(restart_think_text, add_special_tokens=False)
    restart_end_think_text = kwargs.get("restart_end_think_text", "</think>")
    restart_end_think_ids = tokenizer.encode(restart_end_think_text, add_special_tokens=False)
    task_type = str(kwargs.get("task_type", "default")).lower()

    inject_queues = [[] for _ in range(batch_size)]
    restart_triggered = torch.zeros(batch_size, dtype=torch.bool, device=device)
    restart_draft_visible = torch.zeros(batch_size, dtype=torch.bool, device=device)
    restart_chunk_sizes = torch.zeros(batch_size, dtype=torch.long, device=device)
    mask_next_generated_token = torch.zeros(batch_size, dtype=torch.bool, device=device)
    draft_start_pos = [max_prompt_len - len(end_of_thinking_ids) for _ in range(batch_size)]
    draft_soft_embeds = [[] for _ in range(batch_size)]
    draft_student_log_probs = [[] for _ in range(batch_size)]
    restart_draft_visible_end = torch.zeros(batch_size, dtype=torch.long, device=device)
    restart_chunk_positions = [[] for _ in range(batch_size)]
    restart_chunk_log_probs = [[] for _ in range(batch_size)]
    restart_all_soft_embeds = [[] for _ in range(batch_size)]
    restart_token_start_pos = [-1 for _ in range(batch_size)]

    unfinished_idx = list(range(batch_size))
    past_key_values = None

    def _clone_single_past_key_values(cache, batch_idx):
        if cache is None:
            return None
        single_cache = copy.deepcopy(cache)
        keep_idx = torch.tensor([batch_idx], dtype=torch.long, device=device)
        single_cache.batch_select_indices(keep_idx)
        return single_cache

    def _cached_soft_teacher_reverse_kl(
        teacher_inputs_embeds,
        target_ids,
        student_token_log_probs,
        sample_attention_mask,
        prefix_len,
        batch_idx,
    ):
        if past_key_values is None:
            return None
        if teacher_inputs_embeds.size(1) == 0 or target_ids.size(1) == 0:
            return None
        if teacher_inputs_embeds.size(1) != target_ids.size(1):
            return None
        if student_token_log_probs.size(1) != target_ids.size(1):
            return None

        prefix_cache = _clone_single_past_key_values(past_key_values, batch_idx)
        if prefix_cache is None:
            return None
        if hasattr(prefix_cache, "crop"):
            prefix_cache.crop(prefix_len)

        teacher_attention_mask = sample_attention_mask[
            :,
            : prefix_len + teacher_inputs_embeds.size(1),
        ].clone()
        with torch.no_grad():
            teacher_outputs = model(
                inputs_embeds=teacher_inputs_embeds,
                attention_mask=teacher_attention_mask,
                past_key_values=prefix_cache,
                use_cache=False,
            )

        teacher_log_probs = F.log_softmax(teacher_outputs.logits, dim=-1)
        teacher_token_log_probs = teacher_log_probs.gather(
            2,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)
        token_reverse_kl = student_token_log_probs - teacher_token_log_probs
        return token_reverse_kl.mean().item()

    def _draft_answer_reverse_kl(sample_ids, sample_attention_mask, draft_end, orig, batch_idx):
        draft_answer_start = draft_start_pos[orig] + len(end_of_thinking_ids)
        draft_len = draft_end - draft_answer_start + 1
        if draft_len <= 1:
            return None
        if len(draft_student_log_probs[orig]) < draft_len:
            return None
        if len(draft_soft_embeds[orig]) < draft_len:
            return None

        teacher_inputs_embeds = torch.stack(
            draft_soft_embeds[orig][: draft_len - 1],
            dim=0,
        ).unsqueeze(0)
        target_ids = sample_ids[:, draft_answer_start + 1 : draft_end + 1]
        student_token_log_probs = torch.tensor(
            draft_student_log_probs[orig][1:draft_len],
            dtype=teacher_inputs_embeds.dtype,
            device=device,
        ).unsqueeze(0)
        return _cached_soft_teacher_reverse_kl(
            teacher_inputs_embeds,
            target_ids,
            student_token_log_probs,
            sample_attention_mask,
            draft_answer_start,
            batch_idx,
        )

    def _restart_chunk_reverse_kl(sample_ids, sample_attention_mask, span_start, span_end, orig, batch_idx):
        if span_start <= 0 or span_end < span_start:
            return None

        chunk_len = span_end - span_start + 1
        if len(restart_chunk_log_probs[orig]) < chunk_len:
            return None
        if restart_token_start_pos[orig] < 0:
            return None
        if past_key_values is None:
            return None

        soft_start = (span_start - 1) - restart_token_start_pos[orig]
        soft_end = span_end - restart_token_start_pos[orig]
        if (
            soft_start < 0
            or soft_end > len(restart_all_soft_embeds[orig])
            or soft_end <= soft_start
        ):
            return None

        prefix_len = span_start - 1
        chunk_target_ids = sample_ids[:, span_start : span_end + 1]
        teacher_inputs_embeds = torch.stack(
            restart_all_soft_embeds[orig][soft_start:soft_end],
            dim=0,
        ).unsqueeze(0)
        student_token_log_probs = torch.tensor(
            restart_chunk_log_probs[orig][-chunk_len:],
            dtype=teacher_inputs_embeds.dtype,
            device=device,
        ).unsqueeze(0)
        return _cached_soft_teacher_reverse_kl(
            teacher_inputs_embeds,
            chunk_target_ids,
            student_token_log_probs,
            sample_attention_mask,
            prefix_len,
            batch_idx,
        )

    for step in range(max_new_tokens):
        cur_batch = generated.shape[0]
        if cur_batch == 0:
            break
        unfinished_idx_tensor = torch.tensor(unfinished_idx, dtype=torch.long, device=device)

        if past_key_values is None:
            model_inputs = {"input_ids": generated, "attention_mask": attention_mask}
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attention_mask.dtype, device=device)
            pending_mask_rows = mask_next_generated_token[unfinished_idx_tensor]
            if pending_mask_rows.any():
                attention_mask_new[pending_mask_rows, 0] = 0
                mask_next_generated_token[unfinished_idx_tensor[pending_mask_rows]] = False
            attention_mask = torch.cat([attention_mask, attention_mask_new], dim=1)
            model_inputs = {
                "input_ids": next_tokens.unsqueeze(1),
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values

        next_token_logits = outputs.logits[:, -1, :]
        logits = next_token_logits / temperature
        logits = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)

        probs = F.softmax(logits, dim=-1)
        if do_sample:
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        else:
            next_tokens = torch.argmax(probs, dim=-1)
        raw_token_log_probs = F.log_softmax(next_token_logits, dim=-1).gather(
            1,
            next_tokens.unsqueeze(1),
        ).squeeze(1)
        raw_soft_embeds = torch.matmul(
            F.softmax(next_token_logits, dim=-1),
            embedding_matrix,
        )

        forced_restart_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        for bi, orig in enumerate(unfinished_idx):
            if inject_queues[orig]:
                next_tokens[bi] = inject_queues[orig].pop(0)
                forced_restart_mask[bi] = True

        if (
            task_type == "math"
            and tokenizer.eos_token_id is not None
            and len(restart_end_think_ids) == 1
        ):
            restarted_end_think_mask = (
                restart_triggered[unfinished_idx_tensor]
                & (~forced_restart_mask)
                & (next_tokens == restart_end_think_ids[0])
            )
            if restarted_end_think_mask.any():
                next_tokens[restarted_end_think_mask] = tokenizer.eos_token_id

        for bi, orig in enumerate(unfinished_idx):
            if not restart_triggered[orig] and not forced_restart_mask[bi]:
                draft_student_log_probs[orig].append(raw_token_log_probs[bi].item())
                draft_soft_embeds[orig].append(raw_soft_embeds[bi].detach().clone())

        for bi, orig in enumerate(unfinished_idx):
            all_generated[orig].append(next_tokens[bi].item())
            if (
                restart_triggered[orig].item()
                and (tokenizer.eos_token_id is None or next_tokens[bi].item() != tokenizer.eos_token_id)
            ):
                if restart_token_start_pos[orig] < 0:
                    restart_token_start_pos[orig] = len(all_generated[orig]) - 1
                restart_all_soft_embeds[orig].append(raw_soft_embeds[bi].detach().clone())
                if not forced_restart_mask[bi]:
                    restart_chunk_positions[orig].append(len(all_generated[orig]) - 1)
                    restart_chunk_log_probs[orig].append(raw_token_log_probs[bi].item())

        restart_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        if draft_max_new_tokens > 0:
            for bi, orig in enumerate(unfinished_idx):
                if (
                    restart_triggered[orig]
                    or forced_restart_mask[bi]
                    or (
                        tokenizer.eos_token_id is not None
                        and next_tokens[bi].item() == tokenizer.eos_token_id
                    )
                ):
                    continue
                if len(draft_student_log_probs[orig]) >= draft_max_new_tokens:
                    draft_len = len(draft_student_log_probs[orig])
                    attention_mask[bi, draft_start_pos[orig] :] = 0
                    mask_next_generated_token[orig] = True
                    inject_queues[orig] = list(restart_think_ids)
                    restart_triggered[orig] = True
                    restart_draft_visible[orig] = False
                    restart_chunk_sizes[orig] = max(1, draft_len//4)
                    restart_draft_visible_end[orig] = attention_mask.size(1)
                    restart_chunk_positions[orig] = []
                    restart_chunk_log_probs[orig] = []
                    restart_all_soft_embeds[orig] = []
                    restart_token_start_pos[orig] = -1
                    restart_mask[bi] = True
        if tokenizer.eos_token_id is not None:
            draft_eos_mask = (
                (next_tokens == tokenizer.eos_token_id)
                & (~forced_restart_mask)
                & (~restart_triggered[unfinished_idx_tensor])
            )
            if draft_eos_mask.any():
                for bi in draft_eos_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                    orig = unfinished_idx[bi]
                    draft_end = len(all_generated[orig]) - 2
                    sample_ids = torch.tensor(
                        all_generated[orig],
                        dtype=input_ids.dtype,
                        device=device,
                    ).unsqueeze(0)
                    sample_attention_mask = torch.ones(
                        (1, sample_ids.size(1)),
                        dtype=attention_mask.dtype,
                        device=device,
                    )
                    visible_cols = min(attention_mask.size(1), sample_ids.size(1))
                    sample_attention_mask[:, :visible_cols] = attention_mask[bi : bi + 1, :visible_cols]
                    answer_reverse_kl = _draft_answer_reverse_kl(
                        sample_ids,
                        sample_attention_mask,
                        draft_end,
                        orig,
                        bi,
                    )

                    if answer_reverse_kl is None:
                        continue
                    if answer_reverse_kl > tau_a:
                        draft_len = draft_end - draft_start_pos[orig] + 1
                        attention_mask[bi, draft_start_pos[orig] :] = 0
                        mask_next_generated_token[orig] = True
                        inject_queues[orig] = list(restart_think_ids)
                        restart_triggered[orig] = True
                        restart_draft_visible[orig] = False
                        restart_chunk_sizes[orig] = max(1, draft_len//4)
                        restart_draft_visible_end[orig] = attention_mask.size(1)
                        restart_chunk_positions[orig] = []
                        restart_chunk_log_probs[orig] = []
                        restart_all_soft_embeds[orig] = []
                        restart_token_start_pos[orig] = -1
                        restart_mask[bi] = True

            cur_finished = (
                (next_tokens == tokenizer.eos_token_id)
                & (~forced_restart_mask)
                & (~restart_mask)
            )
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        for bi, orig in enumerate(unfinished_idx):
            if not restart_triggered[orig]:
                continue
            if forced_restart_mask[bi]:
                continue
            if tokenizer.eos_token_id is not None and next_tokens[bi].item() == tokenizer.eos_token_id:
                continue
            chunk_size = int(restart_chunk_sizes[orig].item())
            if chunk_size <= 0 or len(restart_chunk_positions[orig]) < chunk_size:
                continue
            chunk_positions = restart_chunk_positions[orig][-chunk_size:]
            if chunk_positions[-1] - chunk_positions[0] + 1 != chunk_size:
                continue

            sample_ids = torch.tensor(
                all_generated[orig],
                dtype=input_ids.dtype,
                device=device,
            ).unsqueeze(0)
            sample_attention_mask = torch.ones(
                (1, sample_ids.size(1)),
                dtype=attention_mask.dtype,
                device=device,
            )
            visible_cols = min(attention_mask.size(1), sample_ids.size(1))
            sample_attention_mask[:, :visible_cols] = attention_mask[bi : bi + 1, :visible_cols]
            chunk_kl = _restart_chunk_reverse_kl(
                sample_ids,
                sample_attention_mask,
                chunk_positions[0],
                chunk_positions[-1],
                orig,
                bi,
            )
            visible_end = int(restart_draft_visible_end[orig].item())

            if chunk_kl is not None:
                if chunk_kl < tau_r:
                    if visible_end > draft_start_pos[orig]:
                        attention_mask[bi, draft_start_pos[orig] : visible_end] = 1
                    restart_draft_visible[orig] = True
                else:
                    attention_mask[bi, draft_start_pos[orig] : visible_end] = 0
                    restart_draft_visible[orig] = False
            restart_chunk_positions[orig] = []
            restart_chunk_log_probs[orig] = []

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]

        if len(unfinished_idx) == 0:
            break
        generated = generated[keep_idx]
        next_tokens = next_tokens[keep_idx]
        attention_mask = attention_mask[keep_idx]
        if hasattr(past_key_values, "batch_select_indices"):
            past_key_values.batch_select_indices(keep_idx)

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), pad_token_id, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out


# For compatibility with hybrid models such as Qwen3.5 model families
def generate_copt_hybrid(model, tokenizer, **kwargs):

    # ---- **model_inputs ----
    input_ids = kwargs.pop("input_ids")
    attention_mask = kwargs.pop("attention_mask")

    # ---- **gen_kwargs ----
    temperature = kwargs.get("temperature", 1.0)
    top_p = kwargs.get("top_p", 1.0)
    top_k = kwargs.get("top_k", 0)
    min_p = kwargs.get("min_p", 0)
    max_new_tokens = kwargs.get("max_new_tokens", 128)
    do_sample = kwargs.get("do_sample", True)

    batch_size = input_ids.shape[0]
    device = input_ids.device
    embedding_layer = model.get_input_embeddings()
    embedding_matrix = embedding_layer.weight

    end_of_thinking_text = kwargs.get("end_of_thinking_text", "</think>")
    end_of_thinking_ids = tokenizer.encode(end_of_thinking_text, add_special_tokens=False)

    restart_think_text = kwargs.get("restart_think_text", "<think>")
    restart_think_ids = tokenizer.encode(restart_think_text, add_special_tokens=False)

    def _trailing_restart_prefix(prompt_ids):
        ids = prompt_ids.tolist()
        if not restart_think_ids:
            return []
        for start in range(len(ids) - len(restart_think_ids), -1, -1):
            if ids[start : start + len(restart_think_ids)] != restart_think_ids:
                continue
            trailing_text = tokenizer.decode(
                ids[start + len(restart_think_ids) :],
                skip_special_tokens=False,
                clean_up_tokenization_spaces=False,
            )
            if trailing_text.strip() == "":
                return ids[start:]
            break
        return []

    rebuilt_sequences = []
    initial_forced_queues = []
    restart_inject_ids = []
    for i in range(batch_size):
        valid_prompt = input_ids[i][attention_mask[i].bool()]
        restart_prefix = _trailing_restart_prefix(valid_prompt)
        rebuilt_sequences.append(valid_prompt)
        initial_forced_queues.append(list(end_of_thinking_ids))
        restart_inject_ids.append([] if restart_prefix else list(restart_think_ids))

    max_prompt_len = max(seq.size(0) for seq in rebuilt_sequences)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    generated = torch.full((batch_size, max_prompt_len), pad_token_id, dtype=input_ids.dtype, device=device)
    attention_mask = torch.zeros((batch_size, max_prompt_len), dtype=attention_mask.dtype, device=device)
    for i, seq in enumerate(rebuilt_sequences):
        seq_len = seq.size(0)
        generated[i, -seq_len:] = seq
        attention_mask[i, -seq_len:] = 1
    all_generated = [generated[i].clone().tolist() for i in range(batch_size)]

    tau_a = kwargs.get("tau_a", 0)
    tau_r = kwargs.get("tau_r", 0)
    draft_max_new_tokens = kwargs.get("draft_max_new_tokens", 1024)
    restart_end_think_text = kwargs.get("restart_end_think_text", "</think>")
    restart_end_think_ids = tokenizer.encode(restart_end_think_text, add_special_tokens=False)
    task_type = str(kwargs.get("task_type", "default")).lower()

    inject_queues = [[] for _ in range(batch_size)]
    restart_triggered = torch.zeros(batch_size, dtype=torch.bool, device=device)
    restart_draft_visible = torch.zeros(batch_size, dtype=torch.bool, device=device)
    restart_chunk_sizes = torch.zeros(batch_size, dtype=torch.long, device=device)
    mask_next_generated_token = torch.zeros(batch_size, dtype=torch.bool, device=device)
    restart_base_pos = [max_prompt_len for _ in range(batch_size)]
    draft_start_pos = [max_prompt_len for _ in range(batch_size)]
    draft_soft_embeds = [[] for _ in range(batch_size)]
    draft_student_log_probs = [[] for _ in range(batch_size)]
    restart_draft_visible_end = torch.zeros(batch_size, dtype=torch.long, device=device)
    restart_chunk_positions = [[] for _ in range(batch_size)]
    restart_chunk_log_probs = [[] for _ in range(batch_size)]
    restart_all_soft_embeds = [[] for _ in range(batch_size)]
    restart_token_start_pos = [-1 for _ in range(batch_size)]
    linear_cache_state_snapshots = [{} for _ in range(batch_size)]
    prompt_end_logits = [None for _ in range(batch_size)]
    restart_use_prompt_logits = torch.zeros(batch_size, dtype=torch.bool, device=device)

    unfinished_idx = list(range(batch_size))
    past_key_values = None

    def _clone_single_past_key_values(cache, batch_idx):
        return clone_single_cache(cache, batch_idx, device)

    def _save_linear_cache_snapshot(batch_idx, orig, prefix_len):
        if past_key_values is None or prefix_len in linear_cache_state_snapshots[orig]:
            return
        snapshot = snapshot_linear_cache_states(past_key_values, batch_idx, device)
        if snapshot is not None:
            linear_cache_state_snapshots[orig][prefix_len] = snapshot

    def _restore_main_linear_cache_snapshot(batch_idx, orig, prefix_len):
        if past_key_values is None:
            return False
        snapshot = linear_cache_state_snapshots[orig].get(prefix_len)
        if snapshot is None:
            return False
        return restore_linear_cache_states_for_batch(past_key_values, snapshot, batch_idx)

    def _cached_soft_teacher_reverse_kl(
        teacher_inputs_embeds,
        target_ids,
        student_token_log_probs,
        sample_attention_mask,
        prefix_len,
        batch_idx,
        orig,
    ):
        if past_key_values is None:
            return None
        if teacher_inputs_embeds.size(1) == 0 or target_ids.size(1) == 0:
            return None
        if teacher_inputs_embeds.size(1) != target_ids.size(1):
            return None
        if student_token_log_probs.size(1) != target_ids.size(1):
            return None

        prefix_cache = _clone_single_past_key_values(past_key_values, batch_idx)
        if prefix_cache is None:
            return None
        crop_cache(prefix_cache, prefix_len)
        if cache_has_linear_layers(prefix_cache):
            linear_snapshot = linear_cache_state_snapshots[orig].get(prefix_len)
            if linear_snapshot is None:
                return None
            restore_linear_cache_states(prefix_cache, linear_snapshot)

        teacher_attention_mask = sample_attention_mask[
            :,
            : prefix_len + teacher_inputs_embeds.size(1),
        ].clone()
        with torch.no_grad():
            teacher_outputs = model(
                inputs_embeds=teacher_inputs_embeds,
                attention_mask=teacher_attention_mask,
                past_key_values=prefix_cache,
                use_cache=False,
            )

        teacher_log_probs = F.log_softmax(teacher_outputs.logits, dim=-1)
        teacher_token_log_probs = teacher_log_probs.gather(
            2,
            target_ids.unsqueeze(-1),
        ).squeeze(-1)
        token_reverse_kl = student_token_log_probs - teacher_token_log_probs
        return token_reverse_kl.mean().item()

    def _draft_answer_reverse_kl(sample_ids, sample_attention_mask, draft_end, orig, batch_idx):
        draft_answer_start = draft_start_pos[orig] + len(end_of_thinking_ids)
        draft_len = draft_end - draft_answer_start + 1
        if draft_len <= 1:
            return None
        if len(draft_student_log_probs[orig]) < draft_len:
            return None
        if len(draft_soft_embeds[orig]) < draft_len:
            return None

        teacher_inputs_embeds = torch.stack(
            draft_soft_embeds[orig][: draft_len - 1],
            dim=0,
        ).unsqueeze(0)
        target_ids = sample_ids[:, draft_answer_start + 1 : draft_end + 1]
        student_token_log_probs = torch.tensor(
            draft_student_log_probs[orig][1:draft_len],
            dtype=teacher_inputs_embeds.dtype,
            device=device,
        ).unsqueeze(0)
        return _cached_soft_teacher_reverse_kl(
            teacher_inputs_embeds,
            target_ids,
            student_token_log_probs,
            sample_attention_mask,
            draft_answer_start,
            batch_idx,
            orig,
        )

    def _restart_chunk_reverse_kl(sample_ids, sample_attention_mask, span_start, span_end, orig, batch_idx):
        if span_start <= 0 or span_end < span_start:
            return None

        chunk_len = span_end - span_start + 1
        if len(restart_chunk_log_probs[orig]) < chunk_len:
            return None
        if restart_token_start_pos[orig] < 0:
            return None
        if past_key_values is None:
            return None

        soft_start = (span_start - 1) - restart_token_start_pos[orig]
        soft_end = span_end - restart_token_start_pos[orig]
        if (
            soft_start < 0
            or soft_end > len(restart_all_soft_embeds[orig])
            or soft_end <= soft_start
        ):
            return None

        prefix_len = span_start - 1
        chunk_target_ids = sample_ids[:, span_start : span_end + 1]
        teacher_inputs_embeds = torch.stack(
            restart_all_soft_embeds[orig][soft_start:soft_end],
            dim=0,
        ).unsqueeze(0)
        student_token_log_probs = torch.tensor(
            restart_chunk_log_probs[orig][-chunk_len:],
            dtype=teacher_inputs_embeds.dtype,
            device=device,
        ).unsqueeze(0)
        return _cached_soft_teacher_reverse_kl(
            teacher_inputs_embeds,
            chunk_target_ids,
            student_token_log_probs,
            sample_attention_mask,
            prefix_len,
            batch_idx,
            orig,
        )

    total_max_new_tokens = max_new_tokens + max((len(q) for q in initial_forced_queues), default=0)
    for step in range(total_max_new_tokens):
        cur_batch = generated.shape[0]
        if cur_batch == 0:
            break
        unfinished_idx_tensor = torch.tensor(unfinished_idx, dtype=torch.long, device=device)
        processed_masked_rows = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        if past_key_values is None:
            model_inputs = {"input_ids": generated, "attention_mask": attention_mask}
        else:
            attention_mask_new = torch.ones((cur_batch, 1), dtype=attention_mask.dtype, device=device)
            pending_mask_rows = mask_next_generated_token[unfinished_idx_tensor]
            if pending_mask_rows.any():
                attention_mask_new[pending_mask_rows, 0] = 0
                mask_next_generated_token[unfinished_idx_tensor[pending_mask_rows]] = False
                processed_masked_rows = pending_mask_rows.clone()
            attention_mask = torch.cat([attention_mask, attention_mask_new], dim=1)
            model_inputs = {
                "input_ids": next_tokens.unsqueeze(1),
                "past_key_values": past_key_values,
                "attention_mask": attention_mask,
            }

        with torch.no_grad():
            outputs = model(**model_inputs, use_cache=True)
        past_key_values = outputs.past_key_values
        for bi, orig in enumerate(unfinished_idx):
            if prompt_end_logits[orig] is None and len(all_generated[orig]) == restart_base_pos[orig]:
                prompt_end_logits[orig] = outputs.logits[bi, -1, :].detach().clone()
        has_linear_cache = cache_has_linear_layers(past_key_values)
        if has_linear_cache:
            for bi, orig in enumerate(unfinished_idx):
                prefix_len = len(all_generated[orig])
                if (
                    prefix_len == restart_base_pos[orig]
                    or prefix_len == draft_start_pos[orig]
                    or prefix_len == draft_start_pos[orig] + len(end_of_thinking_ids)
                ):
                    _save_linear_cache_snapshot(bi, orig, prefix_len)
            if processed_masked_rows.any():
                for bi in processed_masked_rows.nonzero(as_tuple=False).squeeze(-1).tolist():
                    orig = unfinished_idx[bi]
                    _restore_main_linear_cache_snapshot(bi, orig, restart_base_pos[orig])

        forced_initial_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        forced_restart_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        for bi, orig in enumerate(unfinished_idx):
            if initial_forced_queues[orig]:
                forced_initial_mask[bi] = True

        if forced_initial_mask.all():
            next_tokens = torch.empty((cur_batch,), dtype=input_ids.dtype, device=device)
            for bi, orig in enumerate(unfinished_idx):
                next_tokens[bi] = initial_forced_queues[orig].pop(0)
            raw_token_log_probs = None
            raw_soft_embeds = None
        else:
            next_token_logits = outputs.logits[:, -1, :]
            for bi, orig in enumerate(unfinished_idx):
                if restart_use_prompt_logits[orig] and prompt_end_logits[orig] is not None:
                    next_token_logits[bi] = prompt_end_logits[orig].to(
                        device=next_token_logits.device,
                        dtype=next_token_logits.dtype,
                    )
                    restart_use_prompt_logits[orig] = False
            logits = next_token_logits / temperature
            logits = apply_sampling_filter(logits, top_k=top_k, top_p=top_p, min_p=min_p)
            probs = F.softmax(logits, dim=-1)
            if do_sample:
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
            else:
                next_tokens = torch.argmax(probs, dim=-1)
            raw_token_log_probs = F.log_softmax(next_token_logits, dim=-1).gather(
                1,
                next_tokens.unsqueeze(1),
            ).squeeze(1)
            raw_soft_embeds = torch.matmul(
                F.softmax(next_token_logits, dim=-1),
                embedding_matrix,
            )

            for bi, orig in enumerate(unfinished_idx):
                if initial_forced_queues[orig]:
                    next_tokens[bi] = initial_forced_queues[orig].pop(0)
                    forced_initial_mask[bi] = True
                    continue
                if inject_queues[orig]:
                    next_tokens[bi] = inject_queues[orig].pop(0)
                    forced_restart_mask[bi] = True

        if has_linear_cache:
            for bi, orig in enumerate(unfinished_idx):
                if tokenizer.eos_token_id is not None and next_tokens[bi].item() == tokenizer.eos_token_id:
                    continue
                prefix_len = len(all_generated[orig])
                if forced_restart_mask[bi] and not inject_queues[orig]:
                    _save_linear_cache_snapshot(bi, orig, prefix_len)
                    continue
                if not restart_triggered[orig] or forced_restart_mask[bi]:
                    continue
                chunk_size = int(restart_chunk_sizes[orig].item())
                if chunk_size > 0 and len(restart_chunk_positions[orig]) + 1 >= chunk_size:
                    _save_linear_cache_snapshot(bi, orig, prefix_len)

        if (
            not forced_initial_mask.all()
            and task_type == "math"
            and tokenizer.eos_token_id is not None
            and len(restart_end_think_ids) == 1
        ):
            restarted_end_think_mask = (
                restart_triggered[unfinished_idx_tensor]
                & (~forced_restart_mask)
                & (~forced_initial_mask)
                & (next_tokens == restart_end_think_ids[0])
            )
            if restarted_end_think_mask.any():
                next_tokens[restarted_end_think_mask] = tokenizer.eos_token_id

        for bi, orig in enumerate(unfinished_idx):
            if (
                not forced_initial_mask[bi]
                and not restart_triggered[orig]
                and not forced_restart_mask[bi]
            ):
                draft_student_log_probs[orig].append(raw_token_log_probs[bi].item())
                draft_soft_embeds[orig].append(raw_soft_embeds[bi].detach().clone())

        for bi, orig in enumerate(unfinished_idx):
            all_generated[orig].append(next_tokens[bi].item())
            if (
                restart_triggered[orig].item()
                and (tokenizer.eos_token_id is None or next_tokens[bi].item() != tokenizer.eos_token_id)
            ):
                if restart_token_start_pos[orig] < 0:
                    restart_token_start_pos[orig] = len(all_generated[orig]) - 1
                restart_all_soft_embeds[orig].append(raw_soft_embeds[bi].detach().clone())
                if not forced_restart_mask[bi]:
                    restart_chunk_positions[orig].append(len(all_generated[orig]) - 1)
                    restart_chunk_log_probs[orig].append(raw_token_log_probs[bi].item())

        restart_mask = torch.zeros(cur_batch, dtype=torch.bool, device=device)
        if draft_max_new_tokens > 0:
            for bi, orig in enumerate(unfinished_idx):
                if (
                    forced_initial_mask[bi]
                    or restart_triggered[orig]
                    or forced_restart_mask[bi]
                    or (
                        tokenizer.eos_token_id is not None
                        and next_tokens[bi].item() == tokenizer.eos_token_id
                    )
                ):
                    continue
                if len(draft_student_log_probs[orig]) >= draft_max_new_tokens:
                    draft_len = len(draft_student_log_probs[orig])
                    attention_mask[bi, restart_base_pos[orig] :] = 0
                    mask_next_generated_token[orig] = True
                    inject_queues[orig] = list(restart_inject_ids[orig])
                    if not inject_queues[orig]:
                        restart_use_prompt_logits[orig] = True
                    restart_triggered[orig] = True
                    restart_draft_visible[orig] = False
                    restart_chunk_sizes[orig] = max(1, draft_len//4)
                    restart_draft_visible_end[orig] = attention_mask.size(1)
                    restart_chunk_positions[orig] = []
                    restart_chunk_log_probs[orig] = []
                    restart_all_soft_embeds[orig] = []
                    restart_token_start_pos[orig] = -1
                    _restore_main_linear_cache_snapshot(
                        bi,
                        orig,
                        restart_base_pos[orig],
                    )
                    restart_mask[bi] = True
        if tokenizer.eos_token_id is not None:
            draft_eos_mask = (
                (next_tokens == tokenizer.eos_token_id)
                & (~forced_initial_mask)
                & (~forced_restart_mask)
                & (~restart_triggered[unfinished_idx_tensor])
            )
            if draft_eos_mask.any():
                for bi in draft_eos_mask.nonzero(as_tuple=False).squeeze(-1).tolist():
                    orig = unfinished_idx[bi]
                    draft_end = len(all_generated[orig]) - 2
                    sample_ids = torch.tensor(
                        all_generated[orig],
                        dtype=input_ids.dtype,
                        device=device,
                    ).unsqueeze(0)
                    sample_attention_mask = torch.ones(
                        (1, sample_ids.size(1)),
                        dtype=attention_mask.dtype,
                        device=device,
                    )
                    visible_cols = min(attention_mask.size(1), sample_ids.size(1))
                    sample_attention_mask[:, :visible_cols] = attention_mask[bi : bi + 1, :visible_cols]
                    answer_reverse_kl = _draft_answer_reverse_kl(
                        sample_ids,
                        sample_attention_mask,
                        draft_end,
                        orig,
                        bi,
                    )

                    if answer_reverse_kl is None:
                        continue
                    if answer_reverse_kl > tau_a:
                        draft_len = draft_end - draft_start_pos[orig] + 1
                        attention_mask[bi, restart_base_pos[orig] :] = 0
                        mask_next_generated_token[orig] = True
                        inject_queues[orig] = list(restart_inject_ids[orig])
                        if not inject_queues[orig]:
                            restart_use_prompt_logits[orig] = True
                        restart_triggered[orig] = True
                        restart_draft_visible[orig] = False
                        restart_chunk_sizes[orig] = max(1, draft_len//4)
                        restart_draft_visible_end[orig] = attention_mask.size(1)
                        restart_chunk_positions[orig] = []
                        restart_chunk_log_probs[orig] = []
                        restart_all_soft_embeds[orig] = []
                        restart_token_start_pos[orig] = -1
                        _restore_main_linear_cache_snapshot(
                            bi,
                            orig,
                            restart_base_pos[orig],
                        )
                        restart_mask[bi] = True

            cur_finished = (
                (next_tokens == tokenizer.eos_token_id)
                & (~forced_initial_mask)
                & (~forced_restart_mask)
                & (~restart_mask)
            )
        else:
            cur_finished = torch.zeros(cur_batch, dtype=torch.bool, device=device)

        for bi, orig in enumerate(unfinished_idx):
            if not restart_triggered[orig]:
                continue
            if forced_restart_mask[bi]:
                continue
            if tokenizer.eos_token_id is not None and next_tokens[bi].item() == tokenizer.eos_token_id:
                continue
            chunk_size = int(restart_chunk_sizes[orig].item())
            if chunk_size <= 0 or len(restart_chunk_positions[orig]) < chunk_size:
                continue
            chunk_positions = restart_chunk_positions[orig][-chunk_size:]
            if chunk_positions[-1] - chunk_positions[0] + 1 != chunk_size:
                continue

            sample_ids = torch.tensor(
                all_generated[orig],
                dtype=input_ids.dtype,
                device=device,
            ).unsqueeze(0)
            sample_attention_mask = torch.ones(
                (1, sample_ids.size(1)),
                dtype=attention_mask.dtype,
                device=device,
            )
            visible_cols = min(attention_mask.size(1), sample_ids.size(1))
            sample_attention_mask[:, :visible_cols] = attention_mask[bi : bi + 1, :visible_cols]
            chunk_kl = _restart_chunk_reverse_kl(
                sample_ids,
                sample_attention_mask,
                chunk_positions[0],
                chunk_positions[-1],
                orig,
                bi,
            )
            visible_end = int(restart_draft_visible_end[orig].item())

            if chunk_kl is not None:
                if chunk_kl < tau_r:
                    if visible_end > draft_start_pos[orig]:
                        attention_mask[bi, draft_start_pos[orig] : visible_end] = 1
                    restart_draft_visible[orig] = True
                else:
                    attention_mask[bi, draft_start_pos[orig] : visible_end] = 0
                    restart_draft_visible[orig] = False
            restart_chunk_positions[orig] = []
            restart_chunk_log_probs[orig] = []

        keep_idx = (~cur_finished).nonzero(as_tuple=False).squeeze(-1)
        unfinished_idx = [unfinished_idx[i] for i in keep_idx.tolist()]

        if len(unfinished_idx) == 0:
            break
        generated = generated[keep_idx]
        next_tokens = next_tokens[keep_idx]
        attention_mask = attention_mask[keep_idx]
        past_key_values = batch_select_hybrid_cache(past_key_values, keep_idx)

    maxlen = max(len(g) for g in all_generated)
    out = torch.full((batch_size, maxlen), pad_token_id, dtype=torch.long, device=device)
    for i, ids in enumerate(all_generated):
        out[i, :len(ids)] = torch.tensor(ids, dtype=torch.long, device=device)
    return out


def _model_has_hybrid_cache_structure(model):
    cached = getattr(model, "_copt_general_has_hybrid_cache", None)
    if cached is not None:
        return bool(cached)

    config = getattr(model, "config", None)
    if config is not None:
        for key, value in vars(config).items():
            key_l = str(key).lower()
            if "layer" not in key_l and "attn" not in key_l and "attention" not in key_l:
                continue
            if isinstance(value, (list, tuple)) and any("linear" in str(v).lower() for v in value):
                setattr(model, "_copt_general_has_hybrid_cache", True)
                return True
            if isinstance(value, str) and "linear" in value.lower():
                setattr(model, "_copt_general_has_hybrid_cache", True)
                return True

    base_model = getattr(model, "model", model)
    layers = getattr(base_model, "layers", None)
    if layers is None:
        decoder = getattr(base_model, "decoder", None)
        layers = getattr(decoder, "layers", None)

    if layers is not None:
        linear_attr_names = (
            "linear_attn",
            "linear_attention",
            "linear_attn_layer",
            "linear_attention_layer",
        )
        for layer in layers:
            candidates = [layer]
            for attr in linear_attr_names + ("self_attn",):
                child = getattr(layer, attr, None)
                if child is not None:
                    candidates.append(child)
            try:
                candidates.extend(list(layer.children()))
            except Exception:
                pass
            for module in candidates:
                name = module.__class__.__name__.lower()
                if "linearattention" in name or "linear_attention" in name:
                    setattr(model, "_copt_general_has_hybrid_cache", True)
                    return True

    setattr(model, "_copt_general_has_hybrid_cache", False)
    return False


def generate_copt_general(model, tokenizer, **kwargs):
    if _model_has_hybrid_cache_structure(model):
        return generate_copt_hybrid(model, tokenizer, **kwargs)
    return generate_copt(model, tokenizer, **kwargs)
