import json
import numpy as np
import pandas as pd
# ---------- 可选：把 numpy 结构转换成纯 python ----------
def solution_numpy_to_dict(sol):
    """
    输入(可能含 numpy.array)：
      {"header": np.array([...]),
       "rows":   np.array([np.array([...]), ...])}
    输出：
      {"header": [...], "rows": [[...], ...]}
    """
    header = sol["header"].tolist() if isinstance(sol["header"], np.ndarray) else sol["header"]
    rows = sol["rows"]
    if isinstance(rows, np.ndarray):
        rows = [r.tolist() if isinstance(r, np.ndarray) else r for r in rows]
    return {"header": header, "rows": rows}

# ---------- 从 GT 构建结构（要求 header[0] == "House"） ----------
def _load_solution(gt_solution_dict):
    header = gt_solution_dict["header"]
    assert header and header[0] == "House", "GT header must start with 'House'."
    attrs = header[1:]

    sol = {}
    for row in gt_solution_dict["rows"]:
        h = f"h{int(row[0])}"
        sol[h] = {a: v for a, v in zip(attrs, row[1:])}

    houses = [f"h{i+1}" for i in range(len(gt_solution_dict["rows"]))]
    attributes = {a: sorted({sol[h][a] for h in houses}) for a in attrs}

    # alias（大小写/空格归一）
    attr_alias = {a.strip().lower(): a for a in attrs}
    value_alias = {a: {v.strip().lower(): v for v in attributes[a]} for a in attrs}
    return houses, attributes, sol, attr_alias, value_alias

# ---------- 归一化与转换 ----------
def _hid(idx_str: str) -> str:
    return f"h{int(idx_str)}"

def _norm(attr, value, attr_alias, value_alias):
    a_key = attr.strip().lower()
    if a_key not in attr_alias:
        raise ValueError(f"Unknown attr: {attr}")
    a = attr_alias[a_key]
    v_key = value.strip().lower()
    vmap = value_alias[a]
    if v_key not in vmap:
        raise ValueError(f"Value '{value}' not in domain of '{a}': {list(vmap.values())}")
    return a, vmap[v_key]

def _candidate_to_dict(candidate, attributes, houses, attr_alias, value_alias):
    expect_header = ["House"] + list(attributes.keys())
    if candidate.get("header") != expect_header:
        raise ValueError(f"Header mismatch. expected={expect_header}, got={candidate.get('header')}")

    rows = candidate.get("rows")
    if not isinstance(rows, list) or len(rows) != len(houses):
        raise ValueError("Row count mismatch or rows not a list.")

    cand = {}
    for r in rows:
        if not isinstance(r, list) or len(r) != len(expect_header):
            raise ValueError(f"Row shape invalid: {r}")
        h = _hid(r[0])
        row_map = {}
        for attr_name, raw_val in zip(expect_header[1:], r[1:]):
            a, v = _norm(attr_name, raw_val, attr_alias, value_alias)
            row_map[a] = v
        cand[h] = row_map

    if set(cand.keys()) != set(houses):
        raise ValueError("Houses coverage mismatch.")
    return cand

def _check_uniqueness(cand, attributes):
    N = len(cand)
    for a in attributes:
        col = [cand[h][a] for h in cand]
        if len(set(col)) != N:
            raise ValueError(f"Non-unique values in column '{a}': {col}")

def _diff(cand, gt):
    diffs = []
    for h in gt:
        for a, v in gt[h].items():
            if cand[h][a] != v:
                diffs.append({"house": h, "attr": a, "expected": v, "got": cand[h][a]})
    return diffs

# ---------- 只传 candidate 与 gt ----------
def check_final_solution(candidate_solution: dict, gt_solution: dict):
    """
    两者格式相同：{"header":["House",...], "rows":[["1",...], ...]}
    函数内部用 gt 推导 houses/attributes/alias，转换 candidate 与 gt 并比较。
    返回: {"ok": True/False, "errors": [...]} （无错则 errors 为空）
    """
    try:
        houses, attributes, gt_dict, attr_alias, value_alias = _load_solution(gt_solution)
        cand_dict = _candidate_to_dict(candidate_solution, attributes, houses, attr_alias, value_alias)
        _check_uniqueness(cand_dict, attributes)
        diffs = _diff(cand_dict, gt_dict)
        return not diffs, {"ok": not diffs, "errors": [] if not diffs else [{"diffs_vs_gt": diffs}]}
    except Exception as e:
        return False, {"ok": False, "errors": [str(e)]}

# ---------------- demo ----------------
if __name__ == "__main__":

    df = pd.read_parquet("/Users/zwj/puzzle/puzzles_missing1_seed42.parquet")
    row = df.iloc[0]

    # GT from parquet → numpy → 纯 python
    gt_np = row["solution"]


    # 从 GT 构建 houses/attributes/aliases 以及 dict-of-dicts GT

    # 你的预测解（注意：这是字符串，需要 json.loads）
    gt = solution_numpy_to_dict(gt_np)
    print(gt)

    candidate = {
      "header": ["House","Name","Nationality","BookGenre","Food","Color","Animal"],
      "rows": [
        ["1","Bob","german","mystery","grilled cheese","yellow","dog"],
        ["2","Eric","norwegian","fantasy","stew","blue","fish"],
        ["3","Peter","dane","science fiction","spaghetti","green","cat"],
        ["4","Arnold","swede","biography","stir fry","red","bird"],
        ["5","Alice","brit","romance","pizza","white","horse"]
      ]
    }
    Flag, result=check_final_solution(candidate, gt)
    print(Flag)
    print(json.dumps(result, ensure_ascii=False, indent=2))

