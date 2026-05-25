import json
import os
from collections import defaultdict
from utils.extract import md_tex_filter, extract_figure_boxes
from utils.match import match_gt2pred_simple, match_gt2pred_no_split
from utils.match_quick import match_gt2pred_quick
from utils.merge_tables import merge_tables
from utils.relation_utils import (
    extract_table_truncated_info, detect_pred_merges, compute_relation_f1,
    split_pairs_by_page, _norm_table,
    extract_text_truncated_info, detect_pred_text_merges, _norm_text,
)
# from utils.match_full import match_gt2pred_full, match_gt2pred_textblock_full
from utils.read_files import read_md_file
from utils.data_preprocess import normalized_table, clean_string
from registry.registry import DATASET_REGISTRY
from dataset.recog_dataset import *
import pdb
import Levenshtein
from tqdm import tqdm
from func_timeout import FunctionTimedOut, func_timeout
from loguru import logger
import time
import sys
from pylatexenc.latex2text import LatexNodes2Text
import traceback

def remove_consecutive_duplicates(arr):
    """
    删除数组中连续的重复值，保留非连续的重复值
    
    Args:
        arr: 输入的数组/列表
        
    Returns:
        list: 删除连续重复值后的新列表
    """
    if not arr:
        return []
    
    result = [arr[0]]  # 第一个元素总是保留
    
    for i in range(1, len(arr)):
        if arr[i] != arr[i-1]:  # 只有当前元素与前一个不同时才添加
            result.append(arr[i])
    
    return result


def compute_iou(box_a, box_b):
    """计算两个 [x1, y1, x2, y2] bbox 的 IoU。"""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    intersection = max(0, x_right - x_left) * max(0, y_bottom - y_top)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - intersection

    return intersection / union if union > 0 else 0


@DATASET_REGISTRY.register("end2end_dataset")
class End2EndDataset():
    def __init__(self, cfg_task):
        gt_path = cfg_task['dataset']['ground_truth']['data_path']
        pred_folder = cfg_task['dataset']['prediction']['data_path']
        self.match_method = cfg_task['dataset'].get('match_method', 'quick_match')
        filtered_types = cfg_task['dataset'].get('filter')
        self.required_metrics = set(cfg_task.get('metrics', {}).keys())

        with open(gt_path, 'r') as f:
            gt_samples = json.load(f)

        filtered_gt_samples = []
        if filtered_types:
            for gt_sample in gt_samples:
                select_flag = True
                for k, v in filtered_types.items():
                    if gt_sample["page_info"]["page_attribute"][k] != v:
                        select_flag = False
                if select_flag:
                    filtered_gt_samples.append(gt_sample)
        else:
            filtered_gt_samples = gt_samples

        self.samples = self.get_matched_elements(filtered_gt_samples, pred_folder)
        
    def __getitem__(self, cat_name, idx):
        return self.samples[cat_name][idx]
    
    # 处理json中标题中的父亲，儿子关系
    def get_document_heads(self, selected_annos):
        parent_son_list, all_nodes, head_nodes = [], {}, {}
        for item in selected_annos['layout_dets']:
            if item["category_type"] == "title":
                head_nodes[item["anno_id"]] = item
            all_nodes[item["anno_id"]] = item

        for relation in selected_annos["extra"]["relation"]:   # Handle truncated text issues
            if relation["relation_type"] == 'parent_son':
                source_anno_id = relation["source_anno_id"]
                target_anno_id = relation["target_anno_id"]
                if (source_anno_id in head_nodes or source_anno_id == "root") and target_anno_id in head_nodes:
                    parent_son_list.append(relation)
        return parent_son_list, head_nodes

    # 匹配元素 处理文本截断问题，将截断的文本块合并，并将元素按类别存储在字典中
    def get_page_elements(self, selected_annos):

        saved_element_dict = defaultdict(list) #存储元素
        related_truncated = [] #存储需要合并的截断文本块列表
        truncated_all = {} #存储截断文本块信息
        for relation in selected_annos["extra"]["relation"]:   # Handle truncated text issues
            if relation["relation_type"] == 'truncated':
                truncated_all[relation["source_anno_id"]] = ""
                truncated_all[relation["target_anno_id"]] = ""
                exist_flag = False
                for merge_list in related_truncated:
                    if relation["source_anno_id"] in merge_list or relation["target_anno_id"] in merge_list:  # Consider cases where three text blocks may need to be merged
                        merge_list.append(relation["source_anno_id"])
                        merge_list.append(relation["target_anno_id"])
                        exist_flag = True
                if not exist_flag:
                    related_truncated.append([relation["source_anno_id"], relation["target_anno_id"]])       
        
        for item in selected_annos['layout_dets']:
            if item['anno_id'] not in truncated_all.keys():
                saved_element_dict[item["category_type"]].append(item)
            else:
                truncated_all[item['anno_id']] = item
        
        for merge_list in related_truncated:
            merge_list = remove_consecutive_duplicates(merge_list)
            text_block_list = [truncated_all[key] for key in merge_list]
            sorted_block = sorted(text_block_list, key=lambda x: x['order'])
            page_ids = set([block['page_id'] for block in sorted_block])
           
            if sorted_block[0]['category_type'] == 'table':  # merge tabel blocks but keep category as text_block
                if sorted_block[0].get("merged_html") is not None:
                    html = sorted_block[0]["merged_html"]
                else:
                    print(f'{selected_annos["page_info"]["image_path"]} has a merged table group {"_".join([str(block[anno_id]) for block in sorted_block])} with no merged_html annotation.')
                    html = merge_tables([block['html'] for block in sorted_block])
                merged_block = {
                    "category_type": sorted_block[0]["category_type"], # Directly use information from the first block
                    "order": sorted_block[0]["order"],
                    "anno_id": sorted_block[0]["anno_id"],   
                    "html": html,
                    "merge_list": sorted_block,
                    "is_merged": "table",
                    "attribute": {"table_merge_cross_pages": str(len(page_ids))}
                }
            else:  # merge text blocks
                text = ""
                try:
                    for block in sorted_block:
                        text += block.get('text', "")
                except Exception as e:
                    print(sorted_block, selected_annos["page_info"])
                merged_block = {
                    "category_type": sorted_block[0]["category_type"], # Directly use information from the first block
                    "order": sorted_block[0]["order"],
                    "anno_id": sorted_block[0]["anno_id"],   
                    "text": text,
                    "merge_list": sorted_block,
                    "is_merged": "text_block" if sorted_block[0]["category_type"] == "text_block" else "unknown",
                }
                if merged_block.get("is_merged") == "text_block":
                    merged_block["attribute"] = {"text_merge_cross_pages": str(len(page_ids))}
            
            saved_element_dict[sorted_block[0]["category_type"]].append(merged_block)
            # print('Merged truncated')

        return saved_element_dict
    
    # 根据类别列表 category_list 从 gt_page_elements 中提取元素，并将它们合并到一个列表中
    def get_page_elements_list(self, gt_page_elements, category_list):
        element_list = []
        for category_type in category_list:
            if gt_page_elements.get(category_type):
                element_list.extend(gt_page_elements[category_type])
        return element_list

    # 根据元素的 order 字段对元素列表进行排序，并返回排序后的元素列表。
    def get_sorted_text_list(self, selected_annos):
        # txt_type: text, latex, html
        text_list = []
        for item in selected_annos:
            if item.get('order'):
                order = item['order']
            else:
                order = 0
            text_list.append((order, item))
        sorted_text_list = sorted(text_list, key=lambda x: x[0])
        return [_[1] for _ in sorted_text_list]
    
    # 从元素列表 items 中过滤掉 gt_category_type 在 ignore_category_list 中的元素。
    def filtered_out_ignore(self, items, ignore_category_list):
        filted_items = []
        for item in items:
            if item['gt_category_type'] not in ignore_category_list:
                filted_items.append(item)
        return filted_items


    def get_head_paired(self, parent_son_list, head_nodes, md_text_all_list, img_name):
        md_text_all_list.sort(key=lambda x: x['position'][0])
        md_head_list = []
        for item in md_text_all_list:
            if item['content'].strip().startswith('#'):
                md_head_list.append(item['content'].strip())
        return {
            "gt": [parent_son_list, head_nodes],
            "pred": md_head_list,
            "img_id": img_name,
            "edit": 1.0,
        }

    # 计算预测结果和地面真值的阅读顺序之间的编辑距离，并返回包含相关信息的字典。
    def get_order_paired(self, order_match_s, img_name):
        matched = [(item['gt_position'], item['pred_position']) for item in order_match_s if (item['gt_position'] != [""] and item['pred_position'] != "")]
        gt_idx_all = [item['gt_position'] for item in order_match_s if (item['gt_position'] != [""])]
        read_order_pred = [i[0] for i in sorted(matched, key=lambda x: x[1])]  # Sort by pred idx to get Pred ordered GT_idx
        read_order_gt = sum(gt_idx_all, []) # Convert to one-dimensional list
        read_order_gt = [x for x in read_order_gt if x]  # For truncated merges, some discarded classes may be merged in, remove them when calculating edit distance
        gt = sorted(read_order_gt) # Sort by all GT idx to get GT ordered GT_idx
        pred = sum(read_order_pred, [])
        pred = [x for x in pred if x]
        if len(pred) > 0 or len(gt) > 0:
            edit = Levenshtein.distance(gt, pred)/ max(len(pred), len(gt))
            return {
                'gt': gt,  
                'pred': pred,
                'img_id': img_name,
                'edit': edit
            }
        else:
            return {}  # If both GT and pred are empty for the page, return empty

    # 为公式匹配结果添加 img_id 信息。
    def formula_format(self, formula_matches, img_name):
        # formated_list = []
        for i, item in enumerate(formula_matches):
            item["img_id"] = img_name + '_' + str(i)
        return formula_matches

    def get_figure_matched(self, sample, pred_figures, img_name, iou_threshold=0.5):
        """对单个文档进行 GT 与 Pred 的 figure IoU 匹配，计算召回。

        Args:
            sample: GT 文档数据（包含 layout_dets, page_info）
            pred_figures: extract_figure_boxes() 返回的预测图片列表
            img_name: 文档图片名
            iou_threshold: IoU 匹配阈值

        Returns:
            匹配结果 dict，若无 GT figure 则返回 None
        """
        # 提取 GT figures
        gt_figures = []
        page_info = sample.get('page_info', {})
        widths = page_info.get('width', [])
        heights = page_info.get('height', [])

        for det in sample.get('layout_dets', []):
            if det.get('category_type') != 'figure':
                continue
            page_id = det.get('page_id')
            poly = det.get('poly', [])
            if page_id is None or len(poly) < 6:
                continue

            # 检查是否有该页的尺寸信息
            page_idx = page_id - 1
            if page_idx < 0 or page_idx >= len(widths) or page_idx >= len(heights):
                continue
            pw, ph = widths[page_idx], heights[page_idx]
            if not pw or not ph:
                continue

            # poly 格式: [x1,y1, x2,y1, x2,y2, x1,y2] → 归一化到 0-1000
            norm_x1 = poly[0] / pw * 1000
            norm_y1 = poly[1] / ph * 1000
            norm_x2 = poly[4] / pw * 1000
            norm_y2 = poly[5] / ph * 1000
            gt_figures.append({
                "page_id": page_id,
                "bbox_norm": [norm_x1, norm_y1, norm_x2, norm_y2],
                "label": det.get('label', 'figure')
            })

        if not gt_figures:
            return None

        # 按 page_id 分组
        gt_by_page = defaultdict(list)
        for i, fig in enumerate(gt_figures):
            gt_by_page[fig['page_id']].append((i, fig['bbox_norm']))

        pred_by_page = defaultdict(list)
        for j, fig in enumerate(pred_figures):
            pred_by_page[fig['page_id']].append((j, fig['bbox_norm']))

        # 贪心匹配：收集所有 (gt_idx, pred_idx, iou)，按 iou 降序，逐一匹配
        all_pairs = []
        for page_id in gt_by_page:
            for gi, g_box in gt_by_page[page_id]:
                for pi, p_box in pred_by_page.get(page_id, []):
                    iou = compute_iou(g_box, p_box)
                    if iou >= iou_threshold:
                        all_pairs.append((gi, pi, iou))

        all_pairs.sort(key=lambda x: x[2], reverse=True)
        matched_gt = set()
        matched_pred = set()
        matches = []
        for gi, pi, iou in all_pairs:
            if gi not in matched_gt and pi not in matched_pred:
                matched_gt.add(gi)
                matched_pred.add(pi)
                matches.append({"gt_idx": gi, "pred_idx": pi, "iou": iou})

        gt_count = len(gt_figures)
        matched_count = len(matches)

        # 按 figure label 统计 gt_count 和 matched_count
        label_stats = defaultdict(lambda: {'gt_count': 0, 'matched_count': 0})
        for i, fig in enumerate(gt_figures):
            label = fig.get('label', 'figure')
            label_stats[label]['gt_count'] += 1
            if i in matched_gt:
                label_stats[label]['matched_count'] += 1

        return {
            "img_id": img_name,
            "gt_count": gt_count,
            "pred_count": len(pred_figures),
            "matched_count": matched_count,
            "gt": "",
            "pred": "",
            "gt_attribute": [page_info.get('page_attribute', {})],
            "metric": {},
            "label_stats": dict(label_stats),
        }

    def get_table_relation_matched(self, sample, pred_tables_html, img_name, match_threshold=0.15):
        """计算单个文档的表格 truncated 关系 F1 数据。"""
        # 1. 提取 GT truncated pairs
        gt_pairs, groups, anno_map = extract_table_truncated_info(sample)

        # 2. 获取全量 GT 表格，按文档顺序排列
        all_gt_tables = [
            item for item in sample["layout_dets"]
            if item.get("category_type") == "table"
        ]
        all_gt_tables.sort(
            key=lambda x: (x.get("page_id", 0), x.get("order", float("inf")))
        )

        # 3. 如果 GT 无表格或双方都没有 pair 潜力 -> 跳过
        if not all_gt_tables or (not gt_pairs and not pred_tables_html):
            return None

        # 4. 归一化 pred 表格
        pred_norm = [_norm_table(pt["content"]) for pt in pred_tables_html]

        # 5. Pred->GT 跨度搜索
        pred_pairs, pred_match_detail = detect_pred_merges(
            all_gt_tables, pred_norm, match_threshold=match_threshold
        )

        # 6. 如果双方都没有合并对 -> 跳过
        if not gt_pairs and not pred_pairs:
            return None

        # 7. 计算 F1
        metrics = compute_relation_f1(gt_pairs, pred_pairs)

        # 8. 按同页/跨页拆分
        gt_same, gt_cross = split_pairs_by_page(gt_pairs, anno_map)
        pred_same, pred_cross = split_pairs_by_page(pred_pairs, anno_map)
        metrics_same = compute_relation_f1(gt_same, pred_same)
        metrics_cross = compute_relation_f1(gt_cross, pred_cross)

        page_info = sample.get("page_info", {})
        return {
            "img_id": img_name,
            "gt_pairs_count": len(gt_pairs),
            "pred_pairs_count": len(pred_pairs),
            "gt_tables_count": len(all_gt_tables),
            "pred_tables_count": len(pred_tables_html),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            # 同页分项
            "same_page_tp": metrics_same["tp"],
            "same_page_fp": metrics_same["fp"],
            "same_page_fn": metrics_same["fn"],
            # 跨页分项
            "cross_page_tp": metrics_cross["tp"],
            "cross_page_fp": metrics_cross["fp"],
            "cross_page_fn": metrics_cross["fn"],
            "gt": "",
            "pred": "",
            "gt_attribute": [page_info.get("page_attribute", {})],
            "metric": {},
        }

    def get_text_relation_matched(self, sample, pred_text_blocks, img_name, match_threshold=0.15):
        """计算单个文档的段落 truncated 关系 F1 数据。"""
        # 1. 提取 GT truncated pairs
        gt_pairs, groups, anno_map = extract_text_truncated_info(sample)

        # 2. 获取全量 GT text_block，按文档顺序排列
        all_gt_texts = [
            item for item in sample["layout_dets"]
            if item.get("category_type") == "text_block"
        ]
        all_gt_texts.sort(
            key=lambda x: (x.get("page_id", 0), x.get("order", float("inf")))
        )

        # 3. 如果 GT 无文本块或双方都没有 pair 潜力 -> 跳过
        if not all_gt_texts or (not gt_pairs and not pred_text_blocks):
            return None

        # 4. 归一化 pred 文本块
        pred_norm = [_norm_text(pt["content"]) for pt in pred_text_blocks]

        # 5. Pred->GT 跨度搜索
        pred_pairs, pred_match_detail = detect_pred_text_merges(
            all_gt_texts, pred_norm, match_threshold=match_threshold
        )

        # 6. 如果双方都没有合并对 -> 跳过
        if not gt_pairs and not pred_pairs:
            return None

        # 7. 计算 F1
        metrics = compute_relation_f1(gt_pairs, pred_pairs)

        # 8. 按同页/跨页拆分
        gt_same, gt_cross = split_pairs_by_page(gt_pairs, anno_map)
        pred_same, pred_cross = split_pairs_by_page(pred_pairs, anno_map)
        metrics_same = compute_relation_f1(gt_same, pred_same)
        metrics_cross = compute_relation_f1(gt_cross, pred_cross)

        page_info = sample.get("page_info", {})
        return {
            "img_id": img_name,
            "gt_pairs_count": len(gt_pairs),
            "pred_pairs_count": len(pred_pairs),
            "gt_texts_count": len(all_gt_texts),
            "pred_texts_count": len(pred_text_blocks),
            "tp": metrics["tp"],
            "fp": metrics["fp"],
            "fn": metrics["fn"],
            # 同页分项
            "same_page_tp": metrics_same["tp"],
            "same_page_fp": metrics_same["fp"],
            "same_page_fn": metrics_same["fn"],
            # 跨页分项
            "cross_page_tp": metrics_cross["tp"],
            "cross_page_fp": metrics_cross["fp"],
            "cross_page_fn": metrics_cross["fn"],
            "gt": "",
            "pred": "",
            "gt_attribute": [page_info.get("page_attribute", {})],
            "metric": {},
        }

    # 对gt和预测结果进行匹配，调用 process_get_matched_elements 函数进行匹配处理，最终将匹配结果整理成一个字典返回
    def get_matched_elements(self, gt_samples, pred_folder):
        plain_text_match = []
        merged_plain_text_match = []
        display_formula_match = []
        html_table_match = []
        merged_html_table_match = []
        latex_table_match = []
        merged_latex_table_match = []
        order_match = []
        head_match = []
        figure_match = []
        table_relation_match = []
        text_relation_match = []
       
        save_time = time.time()
        process_bar = tqdm(gt_samples, ascii=True, ncols=140)
        img_name_list = [
            'yanbaopptmerge_9081a70ff98b3e7d640660a9412c447d.pdf_1287.jpg'
        ]
        for sample in process_bar:
            img_name = os.path.basename(sample["page_info"]["image_path"])

            # if img_name not in img_name_list:
            #     continue

    
            # print('Process: ', img_name)
            pred_path = os.path.join(pred_folder, img_name[:-4] + '.md')
            if not os.path.exists(pred_path):
                pred_path = os.path.join(pred_folder, img_name[:-4].replace('.pdf', "") + '.mmd')  # nougat
                if not os.path.exists(pred_path):
                    pred_path = os.path.join(pred_folder, img_name[:-4].replace('.pdf', "") + '.md')  # marker
                    if not os.path.exists(pred_path):
                        pred_path = os.path.join(pred_folder, img_name + '.md')
                        if not os.path.exists(pred_path):  # mineru
                            print(f'!!!WARNING: No prediction for {img_name}')
                            continue
      
            process_bar.set_description(f'Processing {os.path.basename(pred_path)}')
            pred_content = read_md_file(pred_path)

            # figure 匹配（在 md_tex_filter 之前提取，因为 md_tex_filter 会去掉图片引用）
            pred_figures = extract_figure_boxes(pred_content)
            figure_sample = self.get_figure_matched(sample, pred_figures, img_name)
            if figure_sample:
                figure_match.append(figure_sample)

            # table relation 匹配（检测 pred 是否正确合并了 truncated 表格）
            pred_dataset_for_relation = md_tex_filter(pred_content)
            pred_tables_html = list(pred_dataset_for_relation.get("html_table", []))
            relation_sample = self.get_table_relation_matched(sample, pred_tables_html, img_name)
            if relation_sample:
                table_relation_match.append(relation_sample)

            # text relation 匹配（检测 pred 是否正确合并了 truncated 段落）
            pred_text_blocks = list(pred_dataset_for_relation.get("text_all", []))
            text_relation_sample = self.get_text_relation_matched(sample, pred_text_blocks, img_name)
            if text_relation_sample:
                text_relation_match.append(text_relation_sample)

            # 如果只需要 figure / table_relation / text_relation 指标，跳过耗时的文本/表格/公式匹配
            if self.required_metrics and self.required_metrics <= {'figure', 'table_relation', 'text_relation'}:
                continue

            # 对单个样本匹配，根据不同的元素类型（如文本块、显示公式、表格等），使用指定的匹配方法将gt与预测结果进行匹配，并返回匹配结果
            result = self.process_get_matched_elements(sample, pred_content, img_name, save_time) # Don't use timeout logic

            [plain_text_match_clean, merged_plain_text_match_clean, formated_display_formula, \
                latex_table_match_s, merged_latex_table_match_s, html_table_match_s, merged_html_table_match_s, \
                order_match_single, head_match_single] = result

            # [plain_text_match_clean, formated_display_formula, latex_table_match_s, html_table_match_s, order_match_single] = result

            # if img_name == 'docstructbench_dianzishu_zhongwenzaixian-o.O-63674848.pdf_101.jpg':
            #     pdb.set_trace()
            if order_match_single:
                order_match.append(order_match_single)
            if head_match_single:
                head_match.append(head_match_single)
            if plain_text_match_clean:
                plain_text_match.extend(plain_text_match_clean)
            if formated_display_formula:
                display_formula_match.extend(formated_display_formula)
            if latex_table_match_s:
                latex_table_match.extend(latex_table_match_s)
            if html_table_match_s:
                html_table_match.extend(html_table_match_s)
            if merged_html_table_match_s:
                merged_html_table_match.extend(merged_html_table_match_s)
            if merged_plain_text_match_clean:
                merged_plain_text_match.extend(merged_plain_text_match_clean)
            if merged_latex_table_match_s:
                merged_latex_table_match.extend(merged_latex_table_match_s)

        # 如果只需要 figure / table_relation / text_relation 指标，直接返回结果，跳过后续文本/表格后处理
        if self.required_metrics and self.required_metrics <= {'figure', 'table_relation', 'text_relation'}:
            matched_samples_all = {
                'figure': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(figure_match),
                'table_relation': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(table_relation_match),
                'text_relation': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(text_relation_match),
            }
            return matched_samples_all

        display_formula_match_clean,display_formula_match_others = [],[]
        for item in display_formula_match:
            pred_category_type = item.get("pred_category_type",None)
            # if pred_category_type == 'equation_inline':
            #     print(item)
            if pred_category_type not in ['equation_inline','equation_isolated', '']:
                gt = item.get('gt',None)
                norm_gt = item.get('norm_gt',None)
                ## latex2unicode
                item['gt'] = LatexNodes2Text().latex_to_text(gt)
                # item['norm_gt'] = LatexNodes2Text().latex_to_text(norm_gt)  # 错了，这里的norm gt是跑的normalized_formula函数，所以再跑latex2unicode会报错
                # 这里的norm_gt应该是跑文本的nrom了
                item['norm_gt'] = clean_string(item['gt'])
                display_formula_match_others.append(item)
            else:
                display_formula_match_clean.append(item)
        display_formula_match = display_formula_match_clean
        if display_formula_match_others and plain_text_match:
            plain_text_match.extend(display_formula_match_others)
            
        #  将latex合并到html 全量428
        if latex_table_match:
            latex_to_html = []
            for latex_table in latex_table_match:
                for k,v in latex_table.items():
                    if 'pred' in k:
                        latex_table[k] = ""
                latex_table['edit'] = 1
                latex_to_html.append(latex_table)
            html_table_match.extend(latex_to_html)
        
        if merged_latex_table_match:
            merged_latex_to_html = []
            for latex_table in merged_latex_table_match:
                for k,v in latex_table.items():
                    if 'pred' in k:
                        latex_table[k] = ""
                latex_table['edit'] = 1
                merged_latex_to_html.append(latex_table)
            merged_html_table_match.extend(merged_latex_to_html)
        
        
        if len(latex_table_match) > len(html_table_match): # Assume model won't randomly output both latex and html, but will choose one
            table_match = latex_table_match
            merged_table_match = merged_latex_table_match
            table_format = 'latex'
        else:
            table_match = html_table_match
            merged_table_match = merged_html_table_match
            table_format = 'html'
            
        # with open('./qwen_latex_table_match.json','w',encoding='utf-8') as f:
        #     json.dump(latex_table_match,f,indent=4,ensure_ascii=False)
        # with open('./qwen_html_table_match.json','w',encoding='utf-8') as f:
        #     json.dump(html_table_match,f,indent=4,ensure_ascii=False)


        matched_samples_all = {
            'text_block': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(plain_text_match),
            "merged_text_block": DATASET_REGISTRY.get('recogition_end2end_base_dataset')(merged_plain_text_match),
            'display_formula':  DATASET_REGISTRY.get('recogition_end2end_base_dataset')(display_formula_match), 
            'table': DATASET_REGISTRY.get('recogition_end2end_table_dataset')(table_match, table_format),
            'merged_table': DATASET_REGISTRY.get('recogition_end2end_table_dataset')(merged_table_match, table_format),
            'reading_order': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(order_match),
            "head": DATASET_REGISTRY.get('recogition_end2end_base_dataset')(head_match),
            'figure': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(figure_match),
            'table_relation': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(table_relation_match),
            'text_relation': DATASET_REGISTRY.get('recogition_end2end_base_dataset')(text_relation_match),
        }
    
        return matched_samples_all

    
    
    #0403 提取gt的table跟pred的table进行匹配 -> 未匹配上的pred_table 去掉html格式然后丢进去混合匹配
    def process_get_matched_elements(self, sample, pred_content, img_name, save_time):
        if self.match_method == 'simple_match':   # add match choice
            match_gt2pred = match_gt2pred_simple
        elif self.match_method == 'quick_match':
            match_gt2pred = match_gt2pred_quick
        elif self.match_method == 'no_split':
            match_gt2pred = match_gt2pred_no_split
        else:
            print('Invalid match method name. The quick_match will be used.')
            match_gt2pred = match_gt2pred_quick

      
        pred_dataset = md_tex_filter(pred_content)
        gt_page_elements = self.get_page_elements(sample)
        parent_son_list, head_nodes = self.get_document_heads(sample)

        gt_mix,pred_dataset_mix = [],[]
        for category in pred_dataset:
            if category not in ['html_table','latex_table','md2html_table']:
                pred_dataset_mix.extend(pred_dataset[category])
        # for category in gt_page_elements:
        #     if category not in ['table']:
        #         gt_mix.extend(gt_page_elements[category])
        gt_mix = self.get_page_elements_list(gt_page_elements, ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption',
                                                'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption',
                                                'header', 'footer', 'page_footnote', 'page_number', 'equation_isolated'])
        if gt_mix:
            gt_mix = self.get_sorted_text_list(gt_mix)

        display_formula_match_s = []
        plain_text_match_clean = []
        merged_plain_text_match_clean = []
        latex_table_match_s = []
        html_table_match_s = []
        merged_latex_table_match_s = []
        merged_html_table_match_s = []
        order_match_single = []
        head_match_single = []
        

        if gt_page_elements.get('table'):
            gt_table = self.get_sorted_text_list(gt_page_elements['table'])
            merged_gt_table = [table_sample for table_sample in self.get_sorted_text_list(gt_page_elements['table']) if table_sample.get('is_merged') == 'table']
            latex_table_len = len(pred_dataset['latex_table']) if pred_dataset['latex_table'] else 0
            html_table_len = len(pred_dataset['html_table']) if pred_dataset['html_table'] else 0
            if latex_table_len == html_table_len and latex_table_len == 0:
                html_table_match_s,unmatch_table_pred = match_gt2pred_simple(gt_table, [], 'html_table', img_name) # Don't consider truncated merging for tables
                html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != [""]]  # Remove extra preds

                merged_html_table_match_s, _ = match_gt2pred_simple(merged_gt_table, [], 'html_table', img_name) # Don't consider truncated merging for tables
                merged_html_table_match_s = [x for x in merged_html_table_match_s if x['gt_idx'] != [""]]  # Remove extra preds

            elif latex_table_len > html_table_len:
                latex_table_match_s, unmatch_table_pred = match_gt2pred_simple(gt_table, pred_dataset['latex_table'], 'latex_table', img_name) # Don't consider truncated merging for tables
                latex_table_match_s = [x for x in latex_table_match_s if x['gt_idx'] != [""]]  # Remove extra preds 

                merged_latex_table_match_s, _ = match_gt2pred_simple(merged_gt_table, pred_dataset['latex_table'], 'latex_table', img_name) # Don't consider truncated merging for tables
                merged_latex_table_match_s = [x for x in merged_latex_table_match_s if x['gt_idx'] != [""]]  # Remove extra preds
            else:
                html_table_match_s,unmatch_table_pred = match_gt2pred_simple(gt_table, pred_dataset['html_table'], 'html_table', img_name) # Don't consider truncated merging for tables
                html_table_match_s = [x for x in html_table_match_s if x['gt_idx'] != [""]]  # Remove extra preds

                merged_html_table_match_s, _ = match_gt2pred_simple(merged_gt_table, pred_dataset['html_table'], 'html_table', img_name) # Don't consider truncated merging for tables
                merged_html_table_match_s = [x for x in merged_html_table_match_s if x['gt_idx'] != [""]]  # Remove extra preds

            if unmatch_table_pred:
                pred_dataset_mix.extend(unmatch_table_pred)

        head_match_single = self.get_head_paired(parent_son_list, head_nodes, pred_dataset_mix, img_name)
        # print(head_match_single)
        
        try:
            match = func_timeout(30, match_gt2pred, args=(gt_mix, pred_dataset_mix, 'text_all', img_name))
        except FunctionTimedOut as e1:
            print(f'Time out for plain text match of {img_name}, match_gt2pred_simple will be used.')
            match,_ = match_gt2pred_simple(gt_mix, pred_dataset_mix, 'text_all', img_name)
        except Exception as e:
            # print(str(e))
            print(traceback.format_exc())
            sys.exit() 

        merged_gt_mix = [gt for gt in gt_mix if gt.get('is_merged') == 'text_block']
        merged_plain_text_match_clean, _ = match_gt2pred_simple(merged_gt_mix, pred_dataset_mix, 'text_all', img_name) # Consider truncated merging for text blocks 
        merged_plain_text_match_clean = [x for x in merged_plain_text_match_clean if x['gt_idx'] != [""]] 

        plain_text_match_s = []
        for item in match:
            gt_category = item.get('gt_category_type',None)
            if gt_category in ['text_block', 'title', 'code_txt', 'code_txt_caption', 'reference', 'equation_caption',
                                                    'figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption',
                                                    'header', 'footer', 'page_footnote', 'page_number']:
                plain_text_match_s.append(item)
            elif gt_category == 'equation_isolated':
                display_formula_match_s.append(item)

        display_formula_match_s = [x for x in display_formula_match_s if x['gt_idx'] != [""]]

        if not plain_text_match_s:
            # print(f'Time out for text match of {img_name}. The plain text match will be empty.')
            # print(f'No text match of {img_name}. The plain text match will be empty.')
            pass
        else:
            # Categories that need to be ignored for text
            plain_text_match_clean = self.filtered_out_ignore(plain_text_match_s, ['figure_caption', 'figure_footnote', 'table_caption', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number', 'equation_caption'])
            # plain_text_match_clean = self.filtered_out_ignore(plain_text_match_s, ['figure_footnote', 'table_footnote', 'code_algorithm', 'code_algorithm_caption', 'header', 'footer', 'page_footnote', 'page_number'])
        order_match_s = plain_text_match_clean
        if order_match_s:
            order_match_single = self.get_order_paired(order_match_s, img_name)


        return [plain_text_match_clean, merged_plain_text_match_clean, display_formula_match_s, latex_table_match_s, merged_latex_table_match_s, html_table_match_s, merged_html_table_match_s, order_match_single, head_match_single]        



@DATASET_REGISTRY.register("recogition_end2end_base_dataset")
class RecognitionEnd2EndBaseDataset():
    def __init__(self, samples):
        img_id = 0
        for sample in samples:
            if not sample.get('img_id'):
                sample['img_id'] = img_id
            img_id += 1
        self.samples = samples
    def __getitem__(self, idx):
        return self.samples[idx]

@DATASET_REGISTRY.register("recogition_end2end_table_dataset")
class RecognitionEnd2EndTableDataset(RecognitionTableDataset):
    def __init__(self, samples, table_format):
        self.pred_table_format = table_format
        self.samples = self.normalize_data(samples)

    def normalize_data(self, samples):
        img_id = 0

        for sample in samples:
            p = sample['pred']
            r = sample['gt']
            p = normalized_table(p, self.pred_table_format)
            r = normalized_table(r)
            sample['norm_gt'] = r
            sample['norm_pred'] = p
            sample['img_id'] = sample['img_id'] if sample.get('img_id') else img_id
            img_id += 1

        return samples
