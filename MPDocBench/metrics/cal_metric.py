# import evaluate
# import random
import json
import os
import time
# from rapidfuzz.distance import Levenshtein
import Levenshtein
from .table_metric import TEDS
import evaluate
import random
from utils.read_files import save_paired_result
from registry.registry import METRIC_REGISTRY
from collections import defaultdict
import pdb
import copy
import pandas as pd
from .cdm_metric import CDM
from .head_metric import HeadTEDS
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import platform

def get_groups(samples, group_info):
    group_samples = defaultdict(list)
    for sample in samples:
        group_samples['all'].append(sample)
        for group in group_info:
            select_flag = True
            for k, v in group.items():
                for gt_attribute in sample['gt_attribute']:   # gt_attribute is a list containing all merged gt attributes
                    if not gt_attribute:   # if no GT attributes, don't include in calculation
                        select_flag = False
                    elif gt_attribute[k] != v:  # if any gt attribute doesn't meet criteria, don't select
                        select_flag = False
            if select_flag:
                group_samples[str(group)].append(sample)
    return group_samples


@METRIC_REGISTRY.register("TEDS")
class call_TEDS():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default'):
        teds = TEDS(structure_only=False)
        teds_structure_only = TEDS(structure_only=True)
        group_scores = defaultdict(list)
        group_scores_structure_only = defaultdict(list)
        samples = self.samples
        per_table_score = {}
        for sample in samples:
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            try:
                score = teds.evaluate(pred, gt)
            except:
                score = 0
                print(f'TEDS score error for table {sample["gt_idx"]} in {sample["img_id"]}. The score is set to 0.')
            try:
                score_structure_only = teds_structure_only.evaluate(pred, gt)
            except:
                score_structure_only = 0
                print(f'TEDS_structure_only score error for table {sample["gt_idx"]} in {sample["img_id"]}. The score is set to 0.')
            # print('TEDS score:', score)
            group_scores['all'].append(score)
            group_scores_structure_only['all'].append(score_structure_only)
            if not sample.get('metric'):
                sample['metric'] = {}
            sample['metric']['TEDS'] = score
            sample['metric']['TEDS_structure_only'] = score_structure_only
            per_table_score[sample['img_id']+'_'+str(sample['gt_idx'])] = {'TEDS': score, 'TEDS_structure_only': score_structure_only}
            for group in group_info:
                select_flag = True
                for k, v in group.items():
                    for gt_attribute in sample['gt_attribute']:   # gt_attribute is a list containing all merged gt attributes
                        if not gt_attribute:   # if no GT attributes, don't include in calculation
                            select_flag = False
                        elif gt_attribute[k] != v:  # if any gt attribute doesn't meet criteria, don't select
                            select_flag = False
                if select_flag:
                    group_scores[str(group)].append(score)
        with open(f'./result/{save_name}_per_table_TEDS.json', 'w', encoding='utf-8') as f:
            json.dump(per_table_score, f, indent=4, ensure_ascii=False)
        result = {}
        for group_name, scores in group_scores.items():
            if len(scores) > 0:
                result[group_name] = sum(scores) / len(scores)    # average of normalized scores at sample level
            else:
                result[group_name] = 'NaN'
                print(f'Warning: Empyty matched samples for {group_name}.')
        
        structure_only_result = {}
        for group_name, scores in group_scores_structure_only.items():
            if len(scores) > 0:
                structure_only_result[group_name] = sum(scores) / len(scores)    # average of normalized scores at sample level
            else:
                structure_only_result[group_name] = 'NaN'
                print(f'Warning: Empyty matched samples for {group_name}.')

        return samples, {'TEDS': result, 'TEDS_structure_only': structure_only_result}


@METRIC_REGISTRY.register("BLEU")
class call_BLEU():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default'):
        group_samples = get_groups(self.samples, group_info)
        result = {}
        for group_name, samples in group_samples.items():
            predictions, references = [], []
            for sample in samples:
                gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
                pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
                predictions.append(gt)
                references.append(pred)
            bleu = evaluate.load("bleu", keep_in_memory=True, experiment_id=random.randint(1,1e8))
            bleu_results = bleu.compute(predictions=predictions, references=references)
            result[group_name] = bleu_results["bleu"]
        
        return self.samples, {'BLEU': result}
    
@METRIC_REGISTRY.register("METEOR")
class call_METEOR():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default'):
        group_samples = get_groups(self.samples, group_info)
        result = {}
        for group_name, samples in group_samples.items():
            predictions, references = [], []
            for sample in samples:
                gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
                pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
                predictions.append(gt)
                references.append(pred)
            meteor = evaluate.load('meteor', keep_in_memory=True, experiment_id=random.randint(1,1e8))
            meteor_results = meteor.compute(predictions=predictions, references=references)
            result[group_name] = meteor_results['meteor']
        
        return self.samples, {'METEOR': result}

@METRIC_REGISTRY.register("Edit_dist")
class call_Edit_dist():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default'):
        samples = self.samples
        for sample in samples:
            img_name = sample['img_id'] if sample['img_id'].endswith('.jpg') or sample['img_id'].endswith('.png') or sample['img_id'].endswith('.pdf') else '_'.join(sample['img_id'].split('_')[:-1])
            sample['image_name'] = img_name
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            upper_len = max(len(pred), len(gt))
            sample['upper_len'] = upper_len
            if len(pred) > 0 or len(gt) > 0:
                edit_dist = Levenshtein.distance(pred, gt)
                if not sample.get('metric'):
                    sample['metric'] = {}
                sample['metric']['Edit_dist'] = edit_dist / upper_len
                sample['Edit_num'] = edit_dist

        if isinstance(samples, list):
            saved_samples = samples
        else:
            saved_samples = samples.samples
        
        if not saved_samples:
            return samples, {'Edit_dist': {'ALL_page_avg': 'NaN'}}

        df = pd.DataFrame(saved_samples)
        up_total_avg = df.groupby("image_name").apply(lambda x: x['Edit_num'].sum() / x['upper_len'].sum()) # page level, sum of edits divided by sum of max(gt,pred) lengths for each sample
        all_total_avg = df['Edit_num'].sum() / df['upper_len'].sum()
        # all_total_avg = df["Edit_dist"].mean()
        per_img_score = up_total_avg.to_dict()
        with open(f'./result/{save_name}_per_page_edit.json', 'w', encoding='utf-8') as f:
            json.dump(per_img_score, f, indent=4, ensure_ascii=False)        
        
        # if 'display_formula' in save_name:
        #     return samples, {'Edit_dist': {'ALL_page_avg': all_total_avg}}
        # else:
        #     return samples, {'Edit_dist': {'ALL_page_avg': up_total_avg.mean()}}
        
        edit_whole = df['Edit_num'].sum() / df['upper_len'].sum()
        df['ratio'] = df['Edit_num'] / df['upper_len']
        edit_sample_avg = df['ratio'].mean()
        # edit_sample_avg = df['metric']['Edit_dist'].mean()
        return samples, {'Edit_dist': {'ALL_page_avg': up_total_avg.mean(), 'edit_whole': edit_whole, 'edit_sample_avg': edit_sample_avg}}
    
def _process_single_cdm_sample(args):
    """Worker function to process a single CDM sample"""
    import re
    idx, sample, output_root, group_info = args
    
    # Create a new CDM instance for this worker to avoid thread safety issues
    cal_cdm = CDM(output_root=output_root)
    
    # Prepare sample data
    sample_copy = copy.deepcopy(sample)
    sample_copy['img_id_cdm'] = str(idx)
    sample_copy['gt'] = sample_copy['gt'].lstrip("$$").rstrip("$$").strip()
    sample_copy['gt'] = sample_copy['gt'].lstrip("$").rstrip("$").strip()
    sample_copy['pred'] = sample_copy['pred'].split("```latex")[-1].split("```")[0]
    sample_copy['pred'] = sample_copy['pred'].lstrip("$$").rstrip("$$").strip()
    sample_copy['pred'] = sample_copy['pred'].lstrip("$").rstrip("$").strip()
    # Strip \[...\] and \(...\) display/inline math delimiters to avoid nested
    # displaymath environments when the template already wraps with \begin{displaymath}
    sample_copy['gt'] = re.sub(r'^\\\[|\\\]$', '', sample_copy['gt'].strip()).strip()
    sample_copy['pred'] = re.sub(r'^\\\[|\\\]$', '', sample_copy['pred'].strip()).strip()
    sample_copy['gt'] = re.sub(r'^\\\(|\\\)$', '', sample_copy['gt'].strip()).strip()
    sample_copy['pred'] = re.sub(r'^\\\(|\\\)$', '', sample_copy['pred'].strip()).strip()
    
    # Calculate CDM score
    cdm_score = cal_cdm.evaluate(sample_copy['gt'], sample_copy['pred'], sample_copy['img_id_cdm'])["F1_score"]
    
    # Add metric to sample
    if not sample_copy.get('metric'):
        sample_copy['metric'] = {}
    sample_copy['metric']['CDM'] = cdm_score
    
    # Check which groups this sample belongs to
    matched_groups = []
    for group in group_info:
        select_flag = True
        for k, v in group.items():
            for gt_attribute in sample_copy['gt_attribute']:
                if not gt_attribute:
                    select_flag = False
                elif gt_attribute[k] != v:
                    select_flag = False
        if select_flag:
            matched_groups.append(str(group))
    
    return {
        'sample': sample_copy,
        'cdm_score': cdm_score,
        'sample_key': sample_copy['img_id'] + '_' + str(sample_copy['gt_idx']),
        'matched_groups': matched_groups,
        'original_index': idx
    }


@METRIC_REGISTRY.register("CDM")
class call_CDM():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default', max_workers=None):
        group_scores = defaultdict(list)
        output_root = f"result/{save_name}/CDM"
        
        if isinstance(self.samples, list):
            original_samples = self.samples
        else:
            original_samples = self.samples.samples
        
        # Prepare arguments for concurrent processing
        worker_args = []
        for idx, sample in enumerate(original_samples):
            worker_args.append((idx, sample, output_root, group_info))

        # Auto-detect a safe number of parallel workers:
        # Each worker spawns xelatex + magick sub-processes, so we keep
        # concurrency moderate to avoid resource exhaustion / process crashes.
        if max_workers is None:
            cpu_count = multiprocessing.cpu_count() or 4
            if platform.system() == 'Darwin':
                # macOS 'spawn' is heavier; be conservative
                max_workers = max(1, min(cpu_count, 8))
            else:
                max_workers = max(1, min(cpu_count * 2, 32))

        # Use concurrent execution
        per_sample_score = {}
        cdm_samples = []
        results = []
        failed_indices = []  # track indices that crash in parallel

        ### PARALLEL VERSION
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_idx = {executor.submit(_process_single_cdm_sample, args): args[0] for args in worker_args}
            
            # Collect results as they complete
            for future in as_completed(future_to_idx):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    idx = future_to_idx[future]
                    print(f'Sample {idx} generated an exception in parallel: {exc}')
                    failed_indices.append(idx)

        ### RETRY failed samples sequentially — avoids false zeros from process crashes
        if failed_indices:
            print(f'[CDM] Retrying {len(failed_indices)} failed samples sequentially ...')
            for idx in failed_indices:
                try:
                    result = _process_single_cdm_sample(worker_args[idx])
                    results.append(result)
                except Exception as exc:
                    print(f'Sample {idx} also failed sequentially: {exc}')
                    sample_copy = copy.deepcopy(original_samples[idx])
                    sample_copy['img_id_cdm'] = str(idx)
                    if not sample_copy.get('metric'):
                        sample_copy['metric'] = {}
                    sample_copy['metric']['CDM'] = 0.0
                    results.append({
                        'sample': sample_copy,
                        'cdm_score': 0.0,
                        'sample_key': sample_copy['img_id'] + '_' + str(sample_copy['gt_idx']),
                        'matched_groups': [],
                        'original_index': idx
                    })
        
        # Sort results by original index to maintain order
        results.sort(key=lambda x: x['original_index'])
        
        # Process results
        for result in results:
            sample = result['sample']
            cdm_score = result['cdm_score']
            sample_key = result['sample_key']
            matched_groups = result['matched_groups']
            
            cdm_samples.append(sample)
            per_sample_score[sample_key] = cdm_score
            group_scores['all'].append(cdm_score)
            
            # Add scores to matched groups
            for group_name in matched_groups:
                group_scores[group_name].append(cdm_score)

        # Save results to files
        with open(f'./result/{save_name}_per_sample_CDM.json', 'w', encoding='utf-8') as f:
            json.dump(per_sample_score, f, indent=4, ensure_ascii=False)

        with open(f'result/{save_name}_result.json', 'w', encoding='utf-8') as f:
            json.dump(cdm_samples, f, indent=4, ensure_ascii=False)

        # Calculate final results
        result = {}
        for group_name, scores in group_scores.items():
            if len(scores) > 0:
                result[group_name] = sum(scores) / len(scores)    # average of normalized scores at sample level
            else:
                result[group_name] = 'NaN'
                print(f'Warning: Empty matched samples for {group_name}.')
        
        return cdm_samples, {'CDM': result}


@METRIC_REGISTRY.register("CDM_plain")
class call_CDM_plain():
    def __init__(self, samples):
        self.samples = samples
    def evaluate(self, group_info=[], save_name='default'):
        if isinstance(self.samples, list):
            cdm_samples = copy.deepcopy(self.samples)
        else:
            cdm_samples = copy.deepcopy(self.samples.samples)
        for idx, sample in enumerate(cdm_samples):
            sample['img_name'] = sample['img_id']
            sample['img_id'] = str(idx)
            sample['gt'] = sample['gt'].lstrip("$$").rstrip("$$").strip()
            sample['pred'] = sample['pred'].split("```latex")[-1].split("```")[0]
            sample['pred'] = sample['pred'].lstrip("$$").rstrip("$$").strip()

        # time_stap = time.time()
        with open(f'result/{save_name}_formula.json', 'w', encoding='utf-8') as f:
            json.dump(cdm_samples, f, indent=4, ensure_ascii=False)
        return self.samples, False


@METRIC_REGISTRY.register("HeadTEDS")
class call_HeadTEDS():
    def __init__(self, samples):
        self.samples = samples

    def evaluate(self, group_info=[], save_name='default'):
        head_teds = HeadTEDS()
        group_scores = defaultdict(list)
        samples = self.samples
        per_sample_score = {}

        for sample in samples:
            gt = sample['norm_gt'] if sample.get('norm_gt') else sample['gt']
            pred = sample['norm_pred'] if sample.get('norm_pred') else sample['pred']
            sample_key = sample['img_id']

            try:
                score, gt_tree_depth = head_teds.evaluate(pred, gt)
            except Exception as e:
                score = 0
                gt_tree_depth = 0
                print(f'HeadTEDS score error for sample {sample_key}: {e}. The score is set to 0.')

            if not sample.get('metric'):
                sample['metric'] = {}
            sample['metric']['HeadTEDS'] = score
            sample["gt_attribute"] = [{"heading tree depth": gt_tree_depth}]
            per_sample_score[sample_key] = score
            group_scores['all'].append(score)

            for group in group_info:
                select_flag = True
                for k, v in group.items():
                    for gt_attribute in sample['gt_attribute']:
                        if not gt_attribute:
                            select_flag = False
                        elif gt_attribute[k] != v:
                            select_flag = False
                if select_flag:
                    group_scores[str(group)].append(score)

        with open(f'./result/{save_name}_per_sample_HeadTEDS.json', 'w', encoding='utf-8') as f:
            json.dump(per_sample_score, f, indent=4, ensure_ascii=False)

        result = {}
        for group_name, scores in group_scores.items():
            if len(scores) > 0:
                result[group_name] = sum(scores) / len(scores)
            else:
                result[group_name] = 'NaN'
                print(f'Warning: Empty matched samples for {group_name}.')

        return samples, {'HeadTEDS': result}



@METRIC_REGISTRY.register("FigureF1")
class call_FigureF1():
    def __init__(self, samples):
        self.samples = samples

    def evaluate(self, group_info=[], save_name='default'):
        """计算文档级图片 F1。"""
        group_scores = defaultdict(list)
        samples = self.samples
        per_doc_score = {}

        for sample in samples:
            gt_count = sample.get('gt_count', 0)
            pred_count = sample.get('pred_count', 0)
            matched_count = sample.get('matched_count', 0)
            precision = matched_count / pred_count if pred_count > 0 else 0
            recall = matched_count / gt_count if gt_count > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

            if not sample.get('metric'):
                sample['metric'] = {}
            sample['metric']['FigureF1'] = f1

            per_doc_score[sample['img_id']] = {
                'f1': f1,
                'precision': precision,
                'recall': recall,
                'gt_count': gt_count,
                'pred_count': pred_count,
                'matched_count': matched_count,
            }

            group_scores['all'].append(f1)

            for group in group_info:
                select_flag = True
                for k, v in group.items():
                    for gt_attribute in sample.get('gt_attribute', []):
                        if not gt_attribute:
                            select_flag = False
                        elif gt_attribute.get(k) != v:
                            select_flag = False
                if select_flag:
                    group_scores[str(group)].append(f1)

        os.makedirs(f'./result/{save_name}', exist_ok=True)
        with open(f'./result/{save_name}_per_doc_FigureF1.json', 'w', encoding='utf-8') as f:
            json.dump(per_doc_score, f, indent=4, ensure_ascii=False)

        result = {}
        for group_name, scores in group_scores.items():
            if len(scores) > 0:
                result[group_name] = sum(scores) / len(scores)
            else:
                result[group_name] = 'NaN'
                print(f'Warning: Empty matched samples for {group_name}.')

        return samples, {'FigureF1': result}


@METRIC_REGISTRY.register("Relation_F1")
class call_RelationF1():
    def __init__(self, samples):
        self.samples = samples

    @staticmethod
    def _f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return p, r, f1

    def evaluate(self, group_info=[], save_name='default'):
        """计算文档级表格截断关系 F1，含同页/跨页分项。"""
        group_scores = defaultdict(list)
        group_scores_same = defaultdict(list)
        group_scores_cross = defaultdict(list)
        samples = self.samples
        per_doc_score = {}

        # Micro 聚合用
        total_tp, total_fp, total_fn = 0, 0, 0
        total_same_tp, total_same_fp, total_same_fn = 0, 0, 0
        total_cross_tp, total_cross_fp, total_cross_fn = 0, 0, 0

        for sample in samples:
            tp = sample.get('tp', 0)
            fp = sample.get('fp', 0)
            fn = sample.get('fn', 0)
            same_tp = sample.get('same_page_tp', 0)
            same_fp = sample.get('same_page_fp', 0)
            same_fn = sample.get('same_page_fn', 0)
            cross_tp = sample.get('cross_page_tp', 0)
            cross_fp = sample.get('cross_page_fp', 0)
            cross_fn = sample.get('cross_page_fn', 0)

            total_tp += tp; total_fp += fp; total_fn += fn
            total_same_tp += same_tp; total_same_fp += same_fp; total_same_fn += same_fn
            total_cross_tp += cross_tp; total_cross_fp += cross_fp; total_cross_fn += cross_fn

            _, _, f1 = self._f1(tp, fp, fn)
            _, _, same_f1 = self._f1(same_tp, same_fp, same_fn)
            _, _, cross_f1 = self._f1(cross_tp, cross_fp, cross_fn)

            if not sample.get('metric'):
                sample['metric'] = {}
            sample['metric']['Relation_F1'] = f1

            per_doc_score[sample['img_id']] = {
                'f1': f1, 'tp': tp, 'fp': fp, 'fn': fn,
                'same_page_f1': same_f1, 'same_page_tp': same_tp, 'same_page_fp': same_fp, 'same_page_fn': same_fn,
                'cross_page_f1': cross_f1, 'cross_page_tp': cross_tp, 'cross_page_fp': cross_fp, 'cross_page_fn': cross_fn,
                'gt_pairs_count': sample.get('gt_pairs_count', 0),
                'pred_pairs_count': sample.get('pred_pairs_count', 0),
            }

            group_scores['all'].append(f1)
            # 只在有该分项 pair 时才计入 macro 平均
            if (same_tp + same_fp + same_fn) > 0:
                group_scores_same['all'].append(same_f1)
            if (cross_tp + cross_fp + cross_fn) > 0:
                group_scores_cross['all'].append(cross_f1)

            for group in group_info:
                select_flag = True
                for k, v in group.items():
                    for gt_attribute in sample.get('gt_attribute', []):
                        if not gt_attribute:
                            select_flag = False
                        elif gt_attribute.get(k) != v:
                            select_flag = False
                if select_flag:
                    group_scores[str(group)].append(f1)
                    if (same_tp + same_fp + same_fn) > 0:
                        group_scores_same[str(group)].append(same_f1)
                    if (cross_tp + cross_fp + cross_fn) > 0:
                        group_scores_cross[str(group)].append(cross_f1)

        # 保存 per-doc 结果
        os.makedirs(f'./result/{save_name}', exist_ok=True)
        with open(f'./result/{save_name}_per_doc_RelationF1.json', 'w', encoding='utf-8') as f:
            json.dump(per_doc_score, f, indent=4, ensure_ascii=False)

        def _agg(group_dict):
            out = {}
            for gn, scores in group_dict.items():
                out[gn] = sum(scores) / len(scores) if scores else 'NaN'
            return out

        # Macro F1（主指标）
        result = _agg(group_scores)
        result_same = _agg(group_scores_same)
        result_cross = _agg(group_scores_cross)

        # Micro F1（辅助参考）
        _, _, micro_f1 = self._f1(total_tp, total_fp, total_fn)
        _, _, micro_same_f1 = self._f1(total_same_tp, total_same_fp, total_same_fn)
        _, _, micro_cross_f1 = self._f1(total_cross_tp, total_cross_fp, total_cross_fn)
        result['micro_f1'] = micro_f1
        result_same['micro_f1'] = micro_same_f1
        result_cross['micro_f1'] = micro_cross_f1

        return samples, {
            'Relation_F1': result,
            'Relation_F1_same_page': result_same,
            'Relation_F1_cross_page': result_cross,
        }