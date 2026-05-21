#!/usr/bin/env python3
"""
Data Cleaning Script - Cleans all data using a simplified regex method and saves the results

Features:
1. Cleans all cases using a simplified regex method.
2. Saves the cleaned data for each case.
3. Ensures the relative order of dicts remains unchanged.
4. Generates a before-and-after cleaning report.
"""

import json
import re
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from collections import Counter
import traceback


@dataclass
class CleanedData:
    """Data structure for cleaned data"""
    case_id: int
    original_type: str  # 'list' or 'str'
    original_length: int
    cleaned_data: List[Dict]
    cleaning_operations: Dict[str, Any]  # Records the cleaning operations performed
    success: bool


class OutputCleaner:
    """Data Cleaner - Based on a simplified regex method"""
    
    def __init__(self):
        # Simplified regular expression patterns
        self.dict_pattern = re.compile(r'\{[^{}]*?"bbox"\s*:\s*\[[^\]]*?\][^{}]*?\}', re.DOTALL)
        self.bbox_pattern = re.compile(r'"bbox"\s*:\s*\[([^\]]+)\]')
        self.missing_delimiter_pattern = re.compile(r'\}\s*\{(?!")')
        
        self.cleaned_results: List[CleanedData] = []
    
    def clean_list_data(self, data: List[Dict], case_id: int) -> CleanedData:
        """Cleans list-type data"""
        
        print(f"🔧 Cleaning List data - Case {case_id}")
        print(f"  Original items: {len(data)}")
        
        cleaned_data = []
        operations = {
            'type': 'list',
            'bbox_fixes': 0,
            'removed_items': 0,
            'original_count': len(data)
        }
        
        for i, item in enumerate(data):
            if not isinstance(item, dict):
                operations['removed_items'] += 1
                continue
                
            # Check the bbox field
            if 'bbox' in item:
                bbox = item['bbox']
                
                # Check bbox length - core logic
                if isinstance(bbox, list) and len(bbox) == 3:
                    print(f"  ⚠️ Item {i}: bbox has only 3 coordinates. Removing bbox, keeping category and text.")
                    # Keep only category and text, ensuring order is preserved
                    new_item = {}
                    if 'category' in item:
                        new_item['category'] = item['category']
                    if 'text' in item:
                        new_item['text'] = item['text']
                    if new_item:  # Add only if there is valid content
                        cleaned_data.append(new_item)
                        operations['bbox_fixes'] += 1
                    else:
                        operations['removed_items'] += 1
                    continue
                elif isinstance(bbox, list) and len(bbox) == 4:
                    # bbox is normal, add directly, preserving original order
                    cleaned_data.append(item.copy())
                    continue
                else:
                    print(f"  ❌ Item {i}: Abnormal bbox format, skipping.")
                    operations['removed_items'] += 1
                    continue
            else:
                # No bbox field, keep if category exists
                if 'category' in item:
                    cleaned_data.append(item.copy())
                    continue
                else:
                    operations['removed_items'] += 1
        
        operations['final_count'] = len(cleaned_data)
        print(f"  ✅ Cleaning complete: {len(cleaned_data)} items, {operations['bbox_fixes']} bbox fixes, {operations['removed_items']} items removed")
        
        return CleanedData(
            case_id=case_id,
            original_type='list',
            original_length=len(data),
            cleaned_data=cleaned_data,
            cleaning_operations=operations,
            success=True
        )
    
    def clean_string_data(self, data_str: str, case_id: int) -> CleanedData:
        """Cleans string-type data"""
        
        print(f"🔧 Cleaning String data - Case {case_id}")
        print(f"  Original length: {len(data_str):,}")
        
        operations = {
            'type': 'str',
            'original_length': len(data_str),
            'delimiter_fixes': 0,
            'tail_truncated': False,
            'truncated_length': 0,
            'duplicate_dicts_removed': 0,
            'final_objects': 0
        }
        
        try:
            # Step 1: Detect and fix missing delimiters
            data_str, delimiter_fixes = self._fix_missing_delimiters(data_str)
            operations['delimiter_fixes'] = delimiter_fixes
            
            # Step 2: Truncate the last incomplete element
            data_str, tail_truncated = self._truncate_last_incomplete_element(data_str)
            operations['tail_truncated'] = tail_truncated
            operations['truncated_length'] = len(data_str)
            
            # Step 3: Remove duplicate complete dict objects, preserving order
            data_str, duplicate_removes = self._remove_duplicate_complete_dicts_preserve_order(data_str)
            operations['duplicate_dicts_removed'] = duplicate_removes
            
            # Step 4: Ensure correct JSON format
            data_str = self._ensure_json_format(data_str)
            
            # Step 5: Try to parse the final result
            final_data = self._parse_final_json(data_str)
            
            if final_data is not None:
                operations['final_objects'] = len(final_data)
                print(f"  ✅ Cleaning complete: {len(final_data)} objects")
                
                return CleanedData(
                    case_id=case_id,
                    original_type='str',
                    original_length=operations['original_length'],
                    cleaned_data=final_data,
                    cleaning_operations=operations,
                    success=True
                )
            else:
                raise Exception("Could not parse the cleaned data")
                
        except Exception as e:
            print(f"  ❌ Cleaning failed: {e}")
            return CleanedData(
                case_id=case_id,
                original_type='str',
                original_length=operations['original_length'],
                cleaned_data=[],
                cleaning_operations=operations,
                success=False
            )
    
    def _fix_missing_delimiters(self, text: str) -> Tuple[str, int]:
        """Fixes missing delimiters"""
        
        fixes = 0
        
        def replace_delimiter(match):
            nonlocal fixes
            fixes += 1
            return '},{'
        
        text = self.missing_delimiter_pattern.sub(replace_delimiter, text)
        
        if fixes > 0:
            print(f"    ✅ Fixed {fixes} missing delimiters")
        
        return text, fixes
    
    def _truncate_last_incomplete_element(self, text: str) -> Tuple[str, bool]:
        """Truncates the last incomplete element"""
        
        # For very long text (>50k) or text not ending with ']', directly truncate the last '{"bbox":'
        needs_truncation = (
            len(text) > 50000 or 
            not text.strip().endswith(']')
        )
        
        if needs_truncation:
            # Check how many dict objects there are
            bbox_count = text.count('{"bbox":')
            
            # If there is only one dict object, do not truncate to avoid deleting the only object
            if bbox_count <= 1:
                print(f"    ⚠️ Only {bbox_count} dict objects found, skipping truncation to avoid deleting all content")
                return text, False
            
            # Find the position of the last '{"bbox":'
            last_bbox_pos = text.rfind('{"bbox":')
            
            if last_bbox_pos > 0:
                # Truncate before this position
                truncated_text = text[:last_bbox_pos].rstrip()
                
                # Remove trailing comma
                if truncated_text.endswith(','):
                    truncated_text = truncated_text[:-1]
                
                print(f"    ✂️ Truncated the last incomplete element, length reduced from {len(text):,} to {len(truncated_text):,}")
                return truncated_text, True
        
        return text, False
    
    def _remove_duplicate_complete_dicts_preserve_order(self, text: str) -> Tuple[str, int]:
        """Removes duplicate complete dict objects, preserving original order"""
        
        # Extract all dict objects, preserving order
        dict_matches = list(self.dict_pattern.finditer(text))
        
        if not dict_matches:
            return text, 0
        
        print(f"    📊 Found {len(dict_matches)} dict objects")
        
        # Deduplication while preserving order: only keep the first occurrence of a dict
        unique_dicts = []
        seen_dict_strings = set()
        total_duplicates = 0
        
        for match in dict_matches:
            dict_str = match.group()
            
            if dict_str not in seen_dict_strings:
                unique_dicts.append(dict_str)
                seen_dict_strings.add(dict_str)
            else:
                total_duplicates += 1
        
        if total_duplicates > 0:
            # Reconstruct the JSON array, preserving the original order
            new_text = '[' + ', '.join(unique_dicts) + ']'
            print(f"    ✅ Removed {total_duplicates} duplicate dicts, keeping {len(unique_dicts)} unique dicts (order preserved)")
            return new_text, total_duplicates
        else:
            print(f"    ✅ No duplicate dict objects found")
            return text, 0
    
    def _ensure_json_format(self, text: str) -> str:
        """Ensures correct JSON format"""
        
        text = text.strip()
        
        if not text.startswith('['):
            text = '[' + text
        
        if not text.endswith(']'):
            # Remove trailing comma
            text = text.rstrip(',').rstrip()
            text += ']'
        
        return text
    
    def _parse_final_json(self, text: str) -> Optional[List[Dict]]:
        """Tries to parse the final JSON"""
        
        try:
            data = json.loads(text)
            if isinstance(data, list):
                return data
        except json.JSONDecodeError as e:
            print(f"    ❌ JSON parsing failed: {e}")
            
            # fallback1: Extract valid dict objects
            valid_dicts = []
            
            for match in self.dict_pattern.finditer(text):
                dict_str = match.group()
                try:
                    dict_obj = json.loads(dict_str)
                    valid_dicts.append(dict_obj)
                except:
                    continue
            
            if valid_dicts:
                print(f"    ✅ Extracted {len(valid_dicts)} valid dicts")
                return valid_dicts
            
            # fallback2: Special handling for a single incomplete dict
            return self._handle_single_incomplete_dict(text)
        
        return None
    
    def _handle_single_incomplete_dict(self, text: str) -> Optional[List[Dict]]:
        """Handles the special case of a single incomplete dict"""
        
        # Check if it's a single incomplete dict case
        if not text.strip().startswith('[{"bbox":'):
            return None
        
        try:
            # Try to extract bbox coordinates
            bbox_match = re.search(r'"bbox"\s*:\s*\[([^\]]+)\]', text)
            if not bbox_match:
                return None
            
            bbox_str = bbox_match.group(1)
            bbox_coords = [int(x.strip()) for x in bbox_str.split(',')]
            
            if len(bbox_coords) != 4:
                return None
            
            # Try to extract category
            category_match = re.search(r'"category"\s*:\s*"([^"]+)"', text)
            category = category_match.group(1) if category_match else "Text"
            
            # Try to extract the beginning of the text (first 10000 characters)
            text_match = re.search(r'"text"\s*:\s*"([^"]{0,10000})', text)
            if text_match:
                text_content = text_match.group(1)
            else:
                text_content = ""
            
            # Construct the fixed dict
            fixed_dict = {
                "bbox": bbox_coords,
                "category": category
            }
            
            if text_content:
                fixed_dict["text"] = text_content
            
            print(f"    🔧 Special fix: single incomplete dict → {fixed_dict}")
            return [fixed_dict]
            
        except Exception as e:
            print(f"    ❌ Special fix failed: {e}")
            return None
    
    def remove_duplicate_category_text_pairs_and_bbox(self, data_list: List[dict], case_id: int) -> List[dict]:
        """Removes duplicate category-text pairs and duplicate bboxes"""
        
        if not data_list or len(data_list) <= 1:
            print(f"    📊 Data length {len(data_list)} <= 1, skipping deduplication check")
            return data_list
        
        print(f"    📊 Original data length: {len(data_list)}")
        
        # 1. Count occurrences and positions of each category-text pair
        category_text_pairs = {}
        for i, item in enumerate(data_list):
            if isinstance(item, dict) and 'category' in item and 'text' in item:
                pair_key = (item.get('category', ''), item.get('text', ''))
                if pair_key not in category_text_pairs:
                    category_text_pairs[pair_key] = []
                category_text_pairs[pair_key].append(i)
        
        # 2. Count occurrences and positions of each bbox
        bbox_pairs = {}
        for i, item in enumerate(data_list):
            if isinstance(item, dict) and 'bbox' in item:
                bbox = item.get('bbox')
                if isinstance(bbox, list) and len(bbox) > 0:
                    bbox_key = tuple(bbox)  # Convert to tuple to use as a dictionary key
                    if bbox_key not in bbox_pairs:
                        bbox_pairs[bbox_key] = []
                    bbox_pairs[bbox_key].append(i)
        
        # 3. Identify items to be removed
        duplicates_to_remove = set()
        
        # 3a. Process category-text pairs that appear 5 or more times
        for pair_key, positions in category_text_pairs.items():
            if len(positions) >= 5:
                category, text = pair_key
                # Keep the first occurrence, remove subsequent duplicates
                positions_to_remove = positions[1:]
                duplicates_to_remove.update(positions_to_remove)
                
                print(f"    🔍 Found duplicate category-text pair: category='{category}', first 50 chars of text='{text[:50]}...'")
                print(f"        Count: {len(positions)}, removing at positions: {positions_to_remove}")
        
        # 3b. Process bboxes that appear 2 or more times
        for bbox_key, positions in bbox_pairs.items():
            if len(positions) >= 2:
                # Keep the first occurrence, remove subsequent duplicates
                positions_to_remove = positions[1:]
                duplicates_to_remove.update(positions_to_remove)
                
                print(f"    🔍 Found duplicate bbox: {list(bbox_key)}")
                print(f"        Count: {len(positions)}, removing at positions: {positions_to_remove}")
        
        if not duplicates_to_remove:
            print(f"    ✅ No category-text pairs or bboxes found exceeding the duplication threshold")
            return data_list
        
        # 4. Remove duplicate items from the original data (preserving order)
        cleaned_data = []
        removed_count = 0
        for i, item in enumerate(data_list):
            if i not in duplicates_to_remove:
                cleaned_data.append(item)
            else:
                removed_count += 1
        
        print(f"    ✅ Deduplication complete: Removed {removed_count} duplicate items")
        print(f"    📊 Cleaned data length: {len(cleaned_data)}")
        
        return cleaned_data

    def clean_model_output(self, model_output: str):
        try:
            # Select cleaning method based on data type
            if isinstance(model_output, list):
                result = self.clean_list_data(model_output, case_id=0)
            else:
                result = self.clean_string_data(str(model_output), case_id=0)
            
            # Add deduplication step: remove duplicate category-text pairs and bboxes
            if result and hasattr(result, 'success') and result.success and result.cleaned_data:
                original_data = result.cleaned_data
                deduplicated_data = self.remove_duplicate_category_text_pairs_and_bbox(original_data, case_id=0)
                # Update the cleaned_data in the CleanedData object
                result.cleaned_data = deduplicated_data
            return result.cleaned_data
        except Exception as e:
            print(f"❌ Case cleaning failed: {e}")
            return model_output
    
    def clean_all_data(self, jsonl_path: str) -> List[CleanedData]:
        """Cleans all data from a JSONL file"""
        
        print(f"🚀 Starting to clean JSONL file: {jsonl_path}")
        
        with open(jsonl_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        datas = []
        for i, line in enumerate(lines):
            if line.strip():
                try:
                    data = json.loads(line)
                    predict_field = data.get('predict')
                    case_id = i + 1
                    
                    print(f"\n{'='*50}")
                    print(f"🎯 Cleaning Case {case_id}")
                    print(f"{'='*50}")
                    
                    # Select cleaning method based on data type
                    if isinstance(predict_field, list):
                        print("📊 Data type: List")
                        result = self.clean_list_data(predict_field, case_id)
                    else:
                        print("📊 Data type: String")
                        result = self.clean_string_data(str(predict_field), case_id)
                    
                    # Add deduplication step: remove duplicate category-text pairs and bboxes
                    if result and hasattr(result, 'success') and result.success and result.cleaned_data:
                        print("🔄 Checking for and removing duplicate category-text pairs and bboxes...")
                        original_data = result.cleaned_data
                        deduplicated_data = self.remove_duplicate_category_text_pairs_and_bbox(original_data, case_id)
                        # Update the cleaned_data in the CleanedData object
                        result.cleaned_data = deduplicated_data
                    data['predict_resized'] = result.cleaned_data

                    datas.append(data)
                    self.cleaned_results.append(result)
                    
                except Exception as e:
                    print(f"❌ Case {i+1} cleaning failed: {e}")
                    traceback.print_exc()
        
        save_path = jsonl_path.replace('.jsonl', '_filtered.jsonl')
        with open(save_path, 'w') as w:
            for data in datas:
                w.write(json.dumps(data, ensure_ascii=False) + '\n')
        print(f"✅ Saved cleaned data to: {save_path}")

        return self.cleaned_results
    
    def save_cleaned_data(self, output_dir: str):
        """Saves the cleaned data"""
        
        print(f"\n💾 Saving cleaned data to: {output_dir}")
        os.makedirs(output_dir, exist_ok=True)
        
        # 1. Save cleaned data for each case
        for result in self.cleaned_results:
            case_filename = f"cleaned_case_{result.case_id:02d}.json"
            case_filepath = os.path.join(output_dir, case_filename)
            
            # Save the cleaned data
            with open(case_filepath, 'w', encoding='utf-8') as f:
                json.dump(result.cleaned_data, f, ensure_ascii=False, indent=2)
            
            print(f"  ✅ Case {result.case_id}: {len(result.cleaned_data)} objects → {case_filename}")
        
        # 2. Save all cleaned data to a single file
        all_cleaned_data = []
        for result in self.cleaned_results:
            all_cleaned_data.append({
                'case_id': result.case_id,
                'original_type': result.original_type,
                'original_length': result.original_length,
                'cleaned_objects_count': len(result.cleaned_data),
                'success': result.success,
                'cleaning_operations': result.cleaning_operations,
                'cleaned_data': result.cleaned_data
            })
        
        all_data_filepath = os.path.join(output_dir, "all_cleaned_data.json")
        with open(all_data_filepath, 'w', encoding='utf-8') as f:
            json.dump(all_cleaned_data, f, ensure_ascii=False, indent=2)
        
        print(f"  📁 All data: {len(all_cleaned_data)} cases → all_cleaned_data.json")
        
        # 3. Generate a cleaning report
        self._generate_cleaning_report(output_dir)
    
    def _generate_cleaning_report(self, output_dir: str):
        """Generates a cleaning report"""
        
        report = []
        report.append("📊 Data Cleaning Report")
        report.append("=" * 60)
        import datetime
        report.append(f"Processing Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("")
        
        # Overall statistics
        total_cases = len(self.cleaned_results)
        successful_cases = sum(1 for r in self.cleaned_results if r.success)
        total_objects = sum(len(r.cleaned_data) for r in self.cleaned_results)
        
        report.append("📈 Overall Statistics:")
        report.append(f"  Total Cases: {total_cases}")
        report.append(f"  Successfully Cleaned: {successful_cases}")
        report.append(f"  Success Rate: {successful_cases/total_cases*100:.1f}%")
        report.append(f"  Total Recovered Objects: {total_objects}")
        report.append("")
        
        # Detailed statistics
        list_results = [r for r in self.cleaned_results if r.original_type == 'list']
        str_results = [r for r in self.cleaned_results if r.original_type == 'str']
        
        if list_results:
            report.append("📋 List Type Cleaning Statistics:")
            for r in list_results:
                ops = r.cleaning_operations
                report.append(f"  Case {r.case_id}: {ops['original_count']} → {ops['final_count']} objects")
                if ops['bbox_fixes'] > 0:
                    report.append(f"    - bbox fixes: {ops['bbox_fixes']}")
                if ops['removed_items'] > 0:
                    report.append(f"    - invalid items removed: {ops['removed_items']}")
            report.append("")
        
        if str_results:
            report.append("📝 String Type Cleaning Statistics:")
            for r in str_results:
                ops = r.cleaning_operations
                status = "✅" if r.success else "❌"
                report.append(f"  Case {r.case_id} {status}: {ops['original_length']:,} chars → {ops['final_objects']} objects")
                details = []
                if ops['delimiter_fixes'] > 0:
                    details.append(f"Delimiter fixes: {ops['delimiter_fixes']}")
                if ops['tail_truncated']:
                    reduction = ops['original_length'] - ops['truncated_length']
                    details.append(f"Tail truncation: -{reduction:,} chars")
                if ops['duplicate_dicts_removed'] > 0:
                    details.append(f"Duplicates removed: {ops['duplicate_dicts_removed']}")
                if details:
                    report.append(f"    - {', '.join(details)}")
            report.append("")
        
        # Note on data order
        report.append("🔄 Data Order Guarantee:")
        report.append("  ✅ The relative order of all dict objects is preserved during cleaning.")
        report.append("  ✅ When deduplicating, the first occurrence of a dict is kept, and subsequent duplicates are removed.")
        report.append("  ✅ The order of items in List-type data is fully preserved.")
        
        # Save the report
        report_filepath = os.path.join(output_dir, "cleaning_report.txt")
        with open(report_filepath, 'w', encoding='utf-8') as f:
            f.write('\n'.join(report))
        
        print(f"  📋 Cleaning report: cleaning_report.txt")
        
        # Also print to console
        print(f"\n{chr(10).join(report)}")


def main():
    """Main function"""
    
    # Create a data cleaner instance
    cleaner = OutputCleaner()
    
    # Input file
    jsonl_path = "output_with_failcase.jsonl"
    
    # Output directory
    output_dir = "output_with_failcase_cleaned"
    
    # Clean all data
    results = cleaner.clean_all_data(jsonl_path)
    
    # Save the cleaned data
    cleaner.save_cleaned_data(output_dir)
    
    print(f"\n🎉 Data cleaning complete!")
    print(f"📁 Cleaned data saved in: {output_dir}")


if __name__ == "__main__":
    main() 