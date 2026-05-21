import os
import io
import re
import json
from tqdm import tqdm
import torch
from concurrent.futures import ThreadPoolExecutor
 

# if torch.version.cuda == '11.8':
#     os.environ["TRITON_PTXAS_PATH"] = "/usr/local/cuda-11.8/bin/ptxas"
os.environ['VLLM_USE_V1'] = '0'
os.environ["CUDA_VISIBLE_DEVICES"] = '0'

os.path.insert('/nas-mmu/zhoubb/code/DeepSeek-OCR-2-main/DeepSeek-OCR2-master/DeepSeek-OCR2-vllm', 0)
from deepseek_ocr2.config import MODEL_PATH, PROMPT, SKIP_REPEAT, MAX_CONCURRENCY, NUM_WORKERS, CROP_MODE

from PIL import Image, ImageDraw, ImageFont
import numpy as np
from vllm import LLM, SamplingParams


llm = LLM(
    model=MODEL_PATH,
    # hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
    # block_size=256,
    # enforce_eager=False,
    trust_remote_code=True, 
    max_model_len=8192,
    swap_space=0,
    max_num_seqs=MAX_CONCURRENCY,
    tensor_parallel_size=1,
    gpu_memory_utilization=0.9,
    # disable_mm_preprocessor_cache=True
)

# logits_processors = [NoRepeatNGramLogitsProcessor(ngram_size=20, window_size=50, whitelist_token_ids= {128821, 128822})] #window for fast；whitelist_token_ids: <td>,</td>

sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=8192,
    # logits_processors=logits_processors,
    skip_special_tokens=False,
    # ignore_eos=False,
    # include_stop_str_in_output=True,
)

def re_match(text):
    pattern = r'(<\|ref\|>(.*?)<\|/ref\|><\|det\|>(.*?)<\|/det\|>)'
    matches = re.findall(pattern, text, re.DOTALL)


    mathes_image = []
    mathes_other = []
    for a_match in matches:
        if '<|ref|>image<|/ref|>' in a_match[0]:
            mathes_image.append(a_match[0])
        else:
            mathes_other.append(a_match[0])
    return matches, mathes_image, mathes_other


def extract_coordinates_and_label(ref_text, image_width, image_height):
    try:
        label_type = ref_text[1]
        cor_list = eval(ref_text[2])
    except Exception as e:
        print(e)
        return None

    return (label_type, cor_list)


def process_single_image(prompt, image):
    """single image"""
    prompt_in = prompt
    cache_item = {
        "prompt": prompt_in,
        "multi_modal_data": {"image": image},
    }
    return cache_item


if __name__ == "__main__":


    INPUT_file = "/nas-mmu/zhoubb/code/MultiIDPBenchmark/normalized_labels/0323_test_data.json"
    OUTPUT_PATH = "/nas-mmu/zhoubb/data/Multi_Page_Documents/markdown/0323/deepseek_ocr2_md"
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    raw_data = json.load(open(INPUT_file))
    data     = [[img_path for (index, img_path, json_path) in v] for k, v in raw_data.items()]
    pdf_names = [k for k in raw_data.keys()]


    prompt = PROMPT

    for pdf_name, img_list in zip(pdf_names, data):
        images = [Image.open(img_path) for img_path in img_list]
    
        with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:  
            batch_inputs = list(tqdm(
                executor.map(process_single_image, [prompt] * len(images), images),
                total=len(images),
                desc="Pre-processed images"
            ))


            outputs_list = llm.generate(
                batch_inputs,
                sampling_params=sampling_params
            )

            md_path = os.path.join(OUTPUT_PATH, pdf_name + '.md')
            contents = ''
            draw_images = []
            jdx = 0
            for output, img in zip(outputs_list, images):
                content = output.outputs[0].text

                # if '<｜end▁of▁sentence｜>' in content: # repeat no eos
                #     content = content.replace('<｜end▁of▁sentence｜>', '')
                # else:
                #     if SKIP_REPEAT:
                #         continue

                # page_num = f'\n<--- Page Split --->'

                matches_ref, matches_images, mathes_other = re_match(content)
                # print(matches_ref)

                for idx, a_match_image in enumerate(matches_images):
                    content = content.replace(a_match_image, f'![](images/' + str(jdx) + '_' + str(idx) + '.jpg)\n')

                for idx, a_match_other in enumerate(mathes_other):
                    content = content.replace(a_match_other, '').replace('\\coloneqq', ':=').replace('\\eqqcolon', '=:').replace('\n\n\n\n', '\n\n').replace('\n\n\n', '\n\n')
                contents += content + "\n\n"
                jdx += 1
            
            if contents[-2:] == "\n\n":
                contents = contents[:-2]

            with open(md_path, 'w', encoding='utf-8') as afile:
                afile.write(contents)



