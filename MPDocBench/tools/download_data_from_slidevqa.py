slidevqa = {
    'pptSlides_eng_26f4ebca': 'https://www.slideshare.net/zschwarz/vertical-apis-as-core-product', 
    'pptSlides_eng_5bb58cde': 'https://www.slideshare.net/mirantis/openstack-architecture-43160012', 
    'pptSlides_eng_bd47b74b': 'https://www.slideshare.net/PhilippeJulio/big-data-architecture', 
    'pptSlides_eng_24541ef2': 'https://www.slideshare.net/dev9105/biochemical-plant-defences', 
    'pptSlides_eng_6d3fdf72': 'https://www.slideshare.net/mpattani/basic-chemistry-52406403', 
    'pptSlides_eng_a04b6aef': 'https://www.slideshare.net/mpattani/chapter-7-the-nervous-system', 
    'pptSlides_eng_52c9fd59': 'https://www.slideshare.net/SurenderRawat3/dna-sequencing-41318444', 
    'pptSlides_eng_82a06813': 'https://www.slideshare.net/RodKing/the-dramatic-s', 
    'pptSlides_eng_1e8e2f71': 'https://www.slideshare.net/saiprasadbagrecha/sap-fibank', 
    'pptSlides_eng_312b78ba': 'https://www.slideshare.net/PatHarlow/regional-key-account-strategy-loral', 
    'pptSlides_eng_f280f799': 'https://www.slideshare.net/ReutersInstitute/tracking-the-future-of-news'
}


import os
import json
from tqdm import tqdm
from datasets import load_dataset


file = "MPDocBench.json"
save_dir = "images"
data = {}
for sample in json.load(open(file)):
    page_info = sample["page_info"]
    pdf_name = page_info["image_path"]
    data[pdf_name] = page_info

# Login using e.g. `huggingface-cli login` to access this dataset
ds = load_dataset("NTT-hil-insight/SlideVQA")
# ds = load_dataset("/Users/zbb/Documents/多页文档解析/SlideVQA")


for split in ["test"]:
    split_ds = ds[split]
    print(f"Processing split: {split}, number of items: {len(split_ds)}")
    for idx, item in tqdm(enumerate(split_ds)):
        for key, url in slidevqa.items():
            pdf_name = os.path.splitext(key)[0]
            if url == item["deck_url"]:
                os.makedirs(os.path.join(save_dir, pdf_name), exist_ok=True)
                for i in range(1, len(data[key+".pdf"]["images_list"])+1):
                    page = item[f"page_{i}"]
                    if page is not None:
                        page.save(os.path.join(save_dir, pdf_name, f"page_{i}.jpg"))
                    else:
                        break