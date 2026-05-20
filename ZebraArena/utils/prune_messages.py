import re
from typing import List, Dict

# === 估算 token 的轻量函数（4 chars ≈ 1 token 的常用经验值）===
def approx_tokens(text: str) -> int:
    if not text:
        return 0
    # 先粗略去掉多余空白，避免空白过多导致误判
    s = re.sub(r"\s+", " ", text)
    return max(1, (len(s) // 4))

def messages_token_count(messages: List[Dict]) -> int:
    return sum(approx_tokens(m.get("content", "")) for m in messages)

# === 删除 <think>…</think> 的压缩函数（保留 query/solution 框架）===
RE_THINK = re.compile(r"<think>.*?</think>", flags=re.DOTALL | re.IGNORECASE)

def strip_think_blocks(text: str) -> str:
    if not text:
        return text
    out = RE_THINK.sub("", text)
    # 折叠多余空白
    out = re.sub(r"\n{3,}", "\n\n", out)
    out = re.sub(r"[ \t]{2,}", " ", out).strip()
    return out

def is_query_like(text: str) -> bool:
    if not text: return False
    # 保留这两类关键信息
    return ("<query>" in text.lower()) or ("environment answer:" in text.lower()) or ("<query_response>" in text.lower())

def prune_messages(
    messages: List[Dict],
    max_input_tokens: int = 12000,
    keep_tail: int = 8,   # 至少保留的尾部消息条数（条，不是轮）
) -> List[Dict]:
    if not messages:
        return messages

    # 1) 先做一次“轻压缩”：去掉历史 assistant 的 <think>…
    compressed: List[Dict] = []
    for i, m in enumerate(messages):
        role = m.get("role", "")
        content = m.get("content", "")
        if role == "assistant":
            content = strip_think_blocks(content)
        compressed.append({"role": role, "content": content})

    # 2) 若在预算内，直接返回
    if messages_token_count(compressed) <= max_input_tokens:
        return compressed

    # 3) 确认“永远保留”的消息：system + 第一条 user（题面）
    must_keep_idx = set()
    if len(compressed) >= 1 and compressed[0].get("role") == "system":
        must_keep_idx.add(0)
    # 第一条 user（通常是题面）
    first_user_idx = None
    for i, m in enumerate(compressed):
        if m.get("role") == "user":
            first_user_idx = i
            break
    if first_user_idx is not None:
        must_keep_idx.add(first_user_idx)

    # 4) 标注“强保留”的消息：包含 <query> / <query_response> / Environment answer
    strong_keep_idx = set()
    for i, m in enumerate(compressed):
        if i in must_keep_idx: 
            continue
        if is_query_like(m.get("content", "")):
            strong_keep_idx.add(i)

    # 5) 基线：保留尾部 keep_tail 条消息
    tail_start = max(0, len(compressed) - keep_tail)
    tail_idx = set(range(tail_start, len(compressed)))

    # 6) 初始保留集合 = must_keep ∪ strong_keep ∪ tail
    keep_idx = set()
    keep_idx |= must_keep_idx
    keep_idx |= strong_keep_idx
    keep_idx |= tail_idx

    # 7) 若仍超预算：从头到尾扫描，尽量删除“非关键且不在尾部”的消息
    def build_kept() -> List[Dict]:
        return [m for i, m in enumerate(compressed) if i in keep_idx]

    def current_tokens() -> int:
        return messages_token_count(build_kept())

    # 如果还是超预算，继续收缩
    if current_tokens() > max_input_tokens:
        # 可删除的候选索引：既不在 must_keep、不在 strong_keep、不在 tail
        deletable = [i for i in range(len(compressed)) if i not in keep_idx]
        for i in deletable:
            keep_idx.discard(i)
            if current_tokens() <= max_input_tokens:
                break

    # 8) 若还超预算：进一步“重度压缩”尾部之外的 assistant 消息（仅保留 JSON 片段）
    if current_tokens() > max_input_tokens:
        def keep_json_blocks(text: str) -> str:
            # 尝试保留 <query> / <solution> / <answer> … 中的 JSON；否则仅保留前 600 字符
            blocks = []
            for tag in ("query", "solution", "answer"):
                for m in re.finditer(fr"<{tag}>(.*?)</{tag}>", text, flags=re.DOTALL | re.IGNORECASE):
                    blocks.append(m.group(0))
            if blocks:
                return "\n\n".join(blocks)
            # 没有结构化块就截断
            return (text[:600] + " …[truncated]") if len(text) > 650 else text

        for i, m in enumerate(compressed):
            if i in keep_idx:
                continue  # 已保留
            role = m.get("role", "")
            if role == "assistant":
                compact = keep_json_blocks(m.get("content", ""))
                compressed[i] = {"role": role, "content": compact}

        # 再尝试删除非关键条，直到达标
        deletable = [i for i in range(len(compressed)) if i not in keep_idx]
        for i in deletable:
            keep_idx.discard(i)
            if current_tokens() <= max_input_tokens:
                break

    # 9) 兜底：如果仍超预算，保底只留 system + 题面 + 尾部若干条
    if messages_token_count(build_kept()) > max_input_tokens:
        minimal_idx = set()
        minimal_idx |= must_keep_idx
        minimal_idx |= tail_idx
        return [m for i, m in enumerate(compressed) if i in minimal_idx]

    # 正常返回
    return build_kept()