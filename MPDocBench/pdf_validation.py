# coding: utf-8
import argparse
import io
import json
import os
import pathlib
import sys
import zipfile
from datetime import datetime, timedelta, timezone
from urllib.parse import quote

import yaml

from registry.registry import DATASET_REGISTRY, EVAL_TASK_REGISTRY
import dataset
import metrics
import task

try:
    import oss2
except ImportError:
    oss2 = None

def process_args(args):
    parser = argparse.ArgumentParser(description='Render latex formulas for comparison.')
    parser.add_argument('--config', '-c', type=str, default='./configs/end2end.yaml')
    parameters = parser.parse_args(args)
    return parameters


def flatten_metric_result(metric_result):
    """将 metric_result.json 展平为 key-value 结构。"""
    flatten_result = {}
    if not isinstance(metric_result, dict):
        return flatten_result

    for task_name, task_payload in metric_result.items():
        if not isinstance(task_payload, dict):
            continue

        all_metrics = task_payload.get("all", {})
        if not isinstance(all_metrics, dict):
            continue

        for metric_name, metric_payload in all_metrics.items():
            metric_prefix = f"{task_name}.{metric_name}"
            if isinstance(metric_payload, dict):
                for field_name, field_value in metric_payload.items():
                    flatten_result[f"{metric_prefix}.{field_name}"] = field_value
            else:
                flatten_result[metric_prefix] = metric_payload

    return flatten_result


def build_cus_outputs(save_dir, project_root):
    metric_path = save_dir / "metric_result.json"
    if not metric_path.exists():
        print(f"[CUS_OUTPUT] 未找到 metric_result.json: {metric_path}")
        return None

    with io.open(metric_path, "r", encoding="utf-8") as f:
        metric_result = json.load(f)

    flattened = flatten_metric_result(metric_result)
    output_path = project_root / "cus_outputs.json"
    with io.open(output_path, "w", encoding="utf-8") as f:
        json.dump(flattened, f, ensure_ascii=False, indent=4)

    print(f"[CUS_OUTPUT] 展平结果已写入: {output_path}")
    return output_path


def zip_save_dir(save_dir):
    zip_path = save_dir.parent / f"{save_dir.name}.zip"
    file_count = 0
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        for file_path in save_dir.rglob("*"):
            if file_path.is_file():
                # 保留 save_name 目录层级
                arcname = file_path.relative_to(save_dir.parent)
                zipf.write(file_path, arcname=str(arcname))
                file_count += 1
    print(f"[ZIP] 打包完成: {zip_path} (文件数: {file_count})")
    return zip_path


def upload_zip_to_oss(zip_path):
    if oss2 is None:
        print("[OSS] 未安装 oss2，无法上传。请先执行: pip install oss2")
        return None
    access_key_id = os.environ.get("OSS_AK")
    access_key_secret = os.environ.get("OSS_SK")
    endpoint = os.environ.get("OSS_ENDPOINT")
    bucket_name = os.environ.get("OSS_BUCKET_NAME")
    object_prefix = os.environ.get("OSS_UPLOAD_PREFIX", "idp_eval").strip("/")

    missing = []
    if not access_key_id:
        missing.append("ALIBABA_CLOUD_ACCESS_KEY_ID")
    if not access_key_secret:
        missing.append("ALIBABA_CLOUD_ACCESS_KEY_SECRET")
    if not endpoint:
        missing.append("OSS_ENDPOINT")
    if not bucket_name:
        missing.append("OSS_BUCKET_NAME")
    if missing:
        print(f"[OSS] 缺少环境变量: {', '.join(missing)}，跳过上传")
        return None

    auth = oss2.Auth(access_key_id, access_key_secret)
    bucket = oss2.Bucket(auth, endpoint, bucket_name)
    object_key = f"{object_prefix}/{zip_path.name}" if object_prefix else zip_path.name

    expires_at = datetime.now(timezone.utc) + timedelta(days=5)
    headers = {
        "Expires": expires_at.strftime("%a, %d %b %Y %H:%M:%S GMT"),
    }

    put_result = bucket.put_object_from_file(object_key, str(zip_path), headers=headers)
    bucket.put_object_acl(object_key, oss2.OBJECT_ACL_PUBLIC_READ)

    endpoint_host = endpoint.replace("https://", "").replace("http://", "")
    public_url = f"https://{bucket_name}.{endpoint_host}/{quote(object_key)}"
    signed_url = bucket.sign_url("GET", object_key, 5 * 24 * 3600)

    print("[OSS] 上传完成")
    print(f"[OSS] HTTP状态码: {put_result.status}")
    print(f"[OSS] Bucket: {bucket_name}")
    print(f"[OSS] ObjectKey: {object_key}")
    print(f"[OSS] 公共访问URL: {public_url}")
    print(f"[OSS] 5天有效签名URL: {signed_url}")

    return {
        "bucket": bucket_name,
        "object_key": object_key,
        "public_url": public_url,
        "signed_url_5d": signed_url,
        "status": put_result.status,
    }

if __name__ == '__main__':
    parameters = process_args(sys.argv[1:])
    config_path = parameters.config
    project_root = pathlib.Path(__file__).resolve().parent
    if isinstance(config_path, (str, pathlib.Path)):
        with io.open(os.path.abspath(config_path), "r", encoding="utf-8") as f:
            cfg = yaml.load(f, Loader=yaml.FullLoader)
    else:
        raise TypeError("Unexpected file type")

    if cfg is not None and not isinstance(cfg, (list, dict, str)):
        raise IOError(  # pragma: no cover
            f"Invalid loaded object type: {type(cfg).__name__}"
        )

    for task in cfg.keys():
        if not cfg.get(task):
            print(f'No config for task {task}')
        dataset_name = cfg[task]['dataset']['dataset_name']
        # metrics_list = [METRIC_REGISTRY.get(i) for i in cfg[task]['metrics']] # TODO: 直接在主函数里实例化
        metrics_list = cfg[task]['metrics']  # 在task里再实例化
        val_dataset = DATASET_REGISTRY.get(dataset_name)(cfg[task])
        val_task = EVAL_TASK_REGISTRY.get(task)
        # val_task(val_dataset, metrics_list)
       
        
        if cfg[task]['dataset']['prediction'].get('data_path'):
            save_name = os.path.basename(cfg[task]['dataset']['prediction']['data_path']) #+ '_' + cfg[task]['dataset'].get('match_method', 'quick_match')
        else:
            save_name = os.path.basename(cfg[task]['dataset']['ground_truth']['data_path']).split('.')[0]
        print('###### Process: ', save_name)
       
        save_dir = project_root / 'result' / save_name
        if not save_dir.exists():
            save_dir.mkdir(parents=True, exist_ok=True)
        if cfg[task]['dataset']['ground_truth'].get('page_info'):
            val_task(val_dataset, metrics_list, cfg[task]['dataset']['ground_truth']['page_info'], save_name)  # 按页面区分
        else:
            val_task(val_dataset, metrics_list, cfg[task]['dataset']['ground_truth']['data_path'], save_name)  # 按页面区分

        # 1) 生成 cus_outputs.json
        # build_cus_outputs(save_dir, project_root)
        # 2) 打包 result/save_name 并上传 OSS（公开 + 5天有效签名链接）
        # zip_path = zip_save_dir(save_dir)
        # upload_zip_to_oss(zip_path)
