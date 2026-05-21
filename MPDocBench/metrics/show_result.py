from collections import defaultdict
from tabulate import tabulate
import pandas as pd
import pdb

def show_result(results):
    for metric_name in results.keys():
        print(f'{metric_name}:')
        score_table = [[k,v] for k,v in results[metric_name].items()]
        print(tabulate(score_table))
        print('='*100)

def sort_nested_dict(d):
    # If it's a dictionary, recursively sort it
    if isinstance(d, dict):
        # Sort the current dictionary
        sorted_dict = {k: sort_nested_dict(v) for k, v in sorted(d.items())}
        return sorted_dict
    # If not a dictionary, return directly
    return d

def get_full_labels_results(samples):
    if not samples:
        return {}
    label_group_dict = defaultdict(lambda: defaultdict(list))
    for sample in samples:
        label_list = []
        if not sample.get("gt_attribute"):
            continue
        for anno in sample["gt_attribute"]:
            for k,v in anno.items():
                label_list.append(k+": "+str(v))
        for label_name in list(set(label_list)):  # Currently if there are merged cases, calculate based on the set of all labels involved after merging
            for metric, score in sample['metric'].items():
                label_group_dict[label_name][metric].append(score)

    print('----Anno Attribute---------------')
    result = {}
    result['sample_count'] = {}
    for attribute in label_group_dict.keys():
        for metric, scores in label_group_dict[attribute].items():
            mean_score = sum(scores) / len(scores)
            if not result.get(metric):
                result[metric] = {}
            result[metric][attribute] = mean_score
            result['sample_count'][attribute] = len(scores)
    result = sort_nested_dict(result)
    show_result(result)
    return result

# def get_page_split(samples, page_info):    # Sample level metric
#     if not page_info:
#         return {}
#     page_split_dict = defaultdict(lambda: defaultdict(list)) 
#     for sample in samples:
#         img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') else '_'.join(sample['img_id'].split('_')[:-1])
#         page_info_s = page_info[img_name]
#         if not sample.get('metric'):
#             continue
#         for metric, score in sample['metric'].items():
#             for k,v in page_info_s.items():
#                 if isinstance(v, list): # special issue
#                     for special_issue in v:
#                         if 'table' not in special_issue:  # Table-related special fields have duplicates
#                             page_split_dict[metric][special_issue].append(score)
#                 else:
#                     page_split_dict[metric][k+": "+str(v)].append(score)
    
#     print('----Page Attribute---------------')
#     result = {}
#     result['sample_count'] = {}
#     for metric in page_split_dict.keys():
#         for attribute, scores in page_split_dict[metric].items():
#             mean_score = sum(scores) / len(scores)
#             if not result.get(metric):
#                 result[metric] = {}
#             result[metric][attribute] = mean_score
#             result['sample_count'][attribute] = len(scores)
#     result = sort_nested_dict(result)
#     show_result(result)
#     return result

def get_figure_label_results(samples):
    """按 figure label（如 印章图、chart、其他图像 等）分组计算 FigureF1。

    对每个 label，汇总所有文档中该 label 的 gt_count 和 matched_count，
    结合文档级 precision 计算 F1。
    - micro: 全局汇总后计算 F1
    - macro: 每文档计算 per-label F1，再跨文档平均
    """
    if not samples:
        return {}

    # 汇总每个 label 的全局统计（micro）和每文档 F1 列表（macro）
    label_totals = defaultdict(lambda: {'gt_count': 0, 'matched_count': 0})
    label_per_doc_f1s = defaultdict(list)
    # 全局统计用于 micro precision
    global_matched = 0
    global_pred = 0

    for sample in samples:
        label_stats = sample.get('label_stats', {})
        doc_matched = sample.get('matched_count', 0)
        doc_pred = sample.get('pred_count', 0)
        doc_precision = doc_matched / doc_pred if doc_pred > 0 else 0

        global_matched += doc_matched
        global_pred += doc_pred

        for label, stats in label_stats.items():
            gt_c = stats.get('gt_count', 0)
            matched_c = stats.get('matched_count', 0)
            label_totals[label]['gt_count'] += gt_c
            label_totals[label]['matched_count'] += matched_c
            if gt_c > 0:
                label_recall = matched_c / gt_c
                label_f1 = 2 * doc_precision * label_recall / (doc_precision + label_recall) if (doc_precision + label_recall) > 0 else 0
                label_per_doc_f1s[label].append(label_f1)

    # 全局 micro precision
    micro_precision = global_matched / global_pred if global_pred > 0 else 0

    # 构建结果：micro F1 和 macro F1
    micro_result = {}
    macro_result = {}
    sample_count = {}
    for label in sorted(label_totals.keys()):
        totals = label_totals[label]
        if totals['gt_count'] > 0:
            micro_recall = totals['matched_count'] / totals['gt_count']
            micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
            micro_result[label] = micro_f1
        else:
            micro_result[label] = 'NaN'
        f1s = label_per_doc_f1s.get(label, [])
        if f1s:
            macro_result[label] = sum(f1s) / len(f1s)
        else:
            macro_result[label] = 'NaN'
        sample_count[label] = totals['gt_count']

    result = {
        'FigureF1_by_label(micro)': micro_result,
        'FigureF1_by_label(macro)': macro_result,
        'gt_count_by_label': sample_count,
    }

    print('----Figure Label Breakdown---------------')
    show_result(result)
    return result


def get_page_split(samples, page_info):   # Page level metric
    if not page_info:
        return {}
    result_list = defaultdict(list)
    for sample in samples:
        img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') or sample['img_id'].endswith('.png') or sample['img_id'].endswith('.pdf') else '_'.join(sample['img_id'].split('_')[:-1])
        page_info_s = page_info[img_name]
        if not sample.get('metric'):
            continue
        for metric, score in sample['metric'].items():
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            result_list[metric].append({
                'image_name': img_name,
                'metric': metric,
                'attribute': 'ALL',
                'score': score,
                'upper_len': max(len(gt), len(pred))
            })
            for k,v in page_info_s.items():
                if isinstance(v, list): # special issue
                    for special_issue in v:
                        if 'table' not in special_issue:  # Table-related special fields have duplicates
                            result_list[metric].append({
                                'image_name': img_name,
                                'metric': metric,
                                'attribute': special_issue,
                                'score': score,
                                'upper_len': max(len(gt), len(pred))
                            })
                else:
                    result_list[metric].append({
                        'image_name': img_name,
                        'metric': metric,
                        'attribute': k+": "+str(v),
                        'score': score,
                        'upper_len': max(len(gt), len(pred))
                    })
    
    # Page level logic, accumulation is only done within pages, and mean operation is performed between pages
    result = {}
    if result_list.get('Edit_dist'):   # 只有Edit_dist需要进行page level的计算
        df = pd.DataFrame(result_list['Edit_dist'])
        up_total_avg = df.groupby(["image_name", "attribute"]).apply(lambda x: (x["score"]*x['upper_len']).sum() / x['upper_len'].sum()).groupby('attribute').mean()  # At page level, accumulate edits, denominator is sum of max(gt, pred) from each sample
        # up_total_avg = df.groupby(["attribute"]).apply(lambda x: (x["score"]*x['upper_len']).sum() / x['upper_len'].sum()) # whole_level
        result['Edit_dist'] = up_total_avg.to_dict()
    for metric in result_list.keys():
        if metric == 'Edit_dist':
            continue
        df = pd.DataFrame(result_list[metric])
        page_avg = df.groupby(["image_name", "attribute"]).apply(lambda x: x["score"].mean()).groupby('attribute').mean() # 页面内部平均以后，再页面间的平均
        result[metric] = page_avg.to_dict()

    result = sort_nested_dict(result)
    # print('----Page Attribute---------------')
    show_result(result)
    return result