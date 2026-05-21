"""
truncated 关系评测的核心算法模块。

提供:
  - UnionFind: 并查集，构建 truncated 连通分量
  - extract_table_truncated_info(): 从 GT 文档中提取表格间 truncated 关系
  - extract_text_truncated_info(): 从 GT 文档中提取段落间 truncated 关系
  - detect_pred_merges(): 从 Pred 侧搜索 GT 表格匹配跨度，推断 pred 合并情况
  - detect_pred_text_merges(): 从 Pred 侧搜索 GT 段落匹配跨度，推断 pred 合并情况
  - compute_relation_f1(): 计算 Precision / Recall / F1
"""

import re
import sys
import io
from collections import defaultdict

import Levenshtein

from utils.merge_tables import normalized_html_table, merge_tables


# ---------------------------------------------------------------------------
# 并查集
# ---------------------------------------------------------------------------

class UnionFind:
    """简单的并查集，用于构建 truncated 表格的连通分量。"""

    def __init__(self):
        self.parent = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[ra] = rb

    def groups(self):
        """返回所有连通分量，每个分量为 set。"""
        result = defaultdict(set)
        for x in self.parent:
            result[self.find(x)].add(x)
        return list(result.values())


# ---------------------------------------------------------------------------
# GT truncated 信息提取
# ---------------------------------------------------------------------------

def extract_table_truncated_info(doc):
    """
    从单个文档中提取表格间 truncated 关系信息。

    Returns:
        gt_pairs: set of (source_anno_id, target_anno_id)
        groups: list of set[int]，每个 set 是一组需要合并的 table anno_id
        anno_map: dict {anno_id: layout_det item}
    """
    anno_map = {}
    for item in doc["layout_dets"]:
        anno_map[item["anno_id"]] = item

    gt_pairs = set()
    uf = UnionFind()

    for relation in doc.get("extra", {}).get("relation", []):
        if relation["relation_type"] != "truncated":
            continue
        src_id = relation["source_anno_id"]
        tgt_id = relation["target_anno_id"]

        src_item = anno_map.get(src_id)
        tgt_item = anno_map.get(tgt_id)
        if not src_item or not tgt_item:
            continue
        if src_item.get("category_type") != "table" or tgt_item.get("category_type") != "table":
            continue

        gt_pairs.add((src_id, tgt_id))
        uf.union(src_id, tgt_id)

    groups = uf.groups() if gt_pairs else []
    return gt_pairs, groups, anno_map


def extract_text_truncated_info(doc):
    """
    从单个文档中提取段落间 truncated 关系信息。

    Returns:
        gt_pairs: set of (source_anno_id, target_anno_id)
        groups: list of set[int]，每个 set 是一组需要合并的 text_block anno_id
        anno_map: dict {anno_id: layout_det item}
    """
    anno_map = {}
    for item in doc["layout_dets"]:
        anno_map[item["anno_id"]] = item

    gt_pairs = set()
    uf = UnionFind()

    for relation in doc.get("extra", {}).get("relation", []):
        if relation["relation_type"] != "truncated":
            continue
        src_id = relation["source_anno_id"]
        tgt_id = relation["target_anno_id"]

        src_item = anno_map.get(src_id)
        tgt_item = anno_map.get(tgt_id)
        if not src_item or not tgt_item:
            continue
        if src_item.get("category_type") != "text_block" or tgt_item.get("category_type") != "text_block":
            continue

        gt_pairs.add((src_id, tgt_id))
        uf.union(src_id, tgt_id)

    groups = uf.groups() if gt_pairs else []
    return gt_pairs, groups, anno_map


# ---------------------------------------------------------------------------
# 归一化 & 编辑距离
# ---------------------------------------------------------------------------

def _norm_table(html):
    """归一化 HTML 表格，用于编辑距离计算。"""
    norm = normalized_html_table(html)
    return norm if norm else html


def _norm_text(text):
    """归一化纯文本，用于编辑距离计算。collapse 空白 + strip。"""
    if not text:
        return ""
    return re.sub(r'\s+', ' ', text).strip()


def _edit_dist(a, b):
    """计算归一化编辑距离。"""
    if not a and not b:
        return 0.0
    if not a or not b:
        return 1.0
    return Levenshtein.distance(a, b) / max(len(a), len(b))


# ---------------------------------------------------------------------------
# Pred->GT 跨度搜索
# ---------------------------------------------------------------------------

def detect_pred_merges(gt_tables_sorted, pred_norm,
                       match_threshold=0.15, candidate_threshold=0.7):
    """
    从 Pred 侧出发，对每个 pred 表格在 GT 表格序列中搜索最佳匹配跨度。

    对于每个 pred 表格，尝试以每个未占用的 GT 表格为起点，逐步向后扩展
    (渐进合并)，找到与该 pred 表格编辑距离最小的 GT 跨度。若最佳跨度
    包含 2+ 个连续 GT 表格，则认为 pred 将它们合并了，产生 merge pair。

    Args:
        gt_tables_sorted: list of GT table layout_det, 按 (page_id, order) 排序
        pred_norm: list[str], 归一化后的 pred 表格
        match_threshold: 最终匹配判定阈值 (默认 0.15)
        candidate_threshold: 预留参数 (默认 0.7)

    Returns:
        pred_pairs: set of (anno_id_a, anno_id_b) -- pred 合并的相邻 GT 表格对
        pred_match_detail: list of dict -- 每个 pred 表格的匹配详情
    """
    n_gt = len(gt_tables_sorted)
    pred_pairs = set()
    pred_match_detail = []
    used_gt = set()  # 已被占用的 GT index

    for pidx, pn in enumerate(pred_norm):
        best_span = None   # (start_idx, end_idx, dist)
        best_dist = float("inf")

        for start in range(n_gt):
            if start in used_gt:
                continue

            htmls = []
            prev_dist = float("inf")

            for end in range(start, min(start + 8, n_gt)):  # 最多尝试合并 8 个
                if end != start and end in used_gt:
                    break

                htmls.append(gt_tables_sorted[end].get("html", ""))

                if len(htmls) == 1:
                    merged_norm = _norm_table(htmls[0])
                else:
                    _stdout = sys.stdout
                    sys.stdout = io.StringIO()
                    try:
                        merged_html = merge_tables(list(htmls))
                    except Exception:
                        merged_html = "".join(htmls)
                    finally:
                        sys.stdout = _stdout
                    merged_norm = _norm_table(merged_html)

                dist = _edit_dist(merged_norm, pn)

                if dist < best_dist:
                    best_dist = dist
                    best_span = (start, end, dist)

                # 渐进合并: 第一次扩展总是尝试 (单个小 GT 距离可能很高)；
                # 从第二个元素开始，距离不再下降则停止。
                if len(htmls) > 1 and dist >= prev_dist:
                    break
                prev_dist = dist

            # 已找到近似完美匹配，提前结束起点搜索
            if best_dist <= 0.005:
                break

        if best_span is not None and best_dist <= match_threshold:
            s, e, d = best_span
            matched_ids = [gt_tables_sorted[i]["anno_id"] for i in range(s, e + 1)]
            for i in range(s, e + 1):
                used_gt.add(i)

            # 跨度 > 1 表示 pred 合并了多个 GT 表格
            if e > s:
                for i in range(s, e):
                    a_id = gt_tables_sorted[i]["anno_id"]
                    b_id = gt_tables_sorted[i + 1]["anno_id"]
                    pred_pairs.add((a_id, b_id))

            pred_match_detail.append({
                "pred_idx": pidx,
                "matched_gt_ids": matched_ids,
                "dist": round(d, 4),
                "is_merge": e > s,
            })
        else:
            pred_match_detail.append({
                "pred_idx": pidx,
                "matched_gt_ids": [],
                "best_dist": round(best_dist, 4) if best_dist < float("inf") else None,
                "is_merge": False,
            })

    return pred_pairs, pred_match_detail


def detect_pred_text_merges(gt_texts_sorted, pred_norm,
                            match_threshold=0.15, candidate_threshold=0.7):
    """
    从 Pred 侧出发，对每个 pred 文本块在 GT 文本块序列中搜索最佳匹配跨度。

    与 detect_pred_merges 逻辑相同，但针对纯文本:
      - GT 内容字段为 'text' 而非 'html'
      - 合并方式为直接拼接而非 merge_tables
      - 归一化使用 _norm_text 而非 _norm_table

    Args:
        gt_texts_sorted: list of GT text_block layout_det, 按 (page_id, order) 排序
        pred_norm: list[str], 归一化后的 pred 文本块
        match_threshold: 最终匹配判定阈值 (默认 0.15)
        candidate_threshold: 预留参数 (默认 0.7)

    Returns:
        pred_pairs: set of (anno_id_a, anno_id_b) -- pred 合并的相邻 GT 文本块对
        pred_match_detail: list of dict -- 每个 pred 文本块的匹配详情
    """
    n_gt = len(gt_texts_sorted)
    pred_pairs = set()
    pred_match_detail = []
    used_gt = set()

    for pidx, pn in enumerate(pred_norm):
        best_span = None
        best_dist = float("inf")

        for start in range(n_gt):
            if start in used_gt:
                continue

            texts = []
            prev_dist = float("inf")

            for end in range(start, min(start + 8, n_gt)):
                if end != start and end in used_gt:
                    break

                texts.append(gt_texts_sorted[end].get("text", ""))
                merged_norm = _norm_text("".join(texts))

                dist = _edit_dist(merged_norm, pn)

                if dist < best_dist:
                    best_dist = dist
                    best_span = (start, end, dist)

                if len(texts) > 1 and dist >= prev_dist:
                    break
                prev_dist = dist

            if best_dist <= 0.005:
                break

        if best_span is not None and best_dist <= match_threshold:
            s, e, d = best_span
            matched_ids = [gt_texts_sorted[i]["anno_id"] for i in range(s, e + 1)]
            for i in range(s, e + 1):
                used_gt.add(i)

            if e > s:
                for i in range(s, e):
                    a_id = gt_texts_sorted[i]["anno_id"]
                    b_id = gt_texts_sorted[i + 1]["anno_id"]
                    pred_pairs.add((a_id, b_id))

            pred_match_detail.append({
                "pred_idx": pidx,
                "matched_gt_ids": matched_ids,
                "dist": round(d, 4),
                "is_merge": e > s,
            })
        else:
            pred_match_detail.append({
                "pred_idx": pidx,
                "matched_gt_ids": [],
                "best_dist": round(best_dist, 4) if best_dist < float("inf") else None,
                "is_merge": False,
            })

    return pred_pairs, pred_match_detail


# ---------------------------------------------------------------------------
# F1 计算
# ---------------------------------------------------------------------------

def split_pairs_by_page(pairs, anno_map):
    """将 pair 集合按 source/target 是否同页拆分。

    Returns:
        same_page_pairs: set, source 与 target 在同一页
        cross_page_pairs: set, source 与 target 在不同页
    """
    same, cross = set(), set()
    for src_id, tgt_id in pairs:
        src_item = anno_map.get(src_id)
        tgt_item = anno_map.get(tgt_id)
        if src_item and tgt_item and src_item.get("page_id") == tgt_item.get("page_id"):
            same.add((src_id, tgt_id))
        else:
            cross.add((src_id, tgt_id))
    return same, cross


def compute_relation_f1(gt_pairs, pred_pairs):
    """计算 Precision, Recall, F1。"""
    tp = len(gt_pairs & pred_pairs)
    fp = len(pred_pairs - gt_pairs)
    fn = len(gt_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"tp": tp, "fp": fp, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
