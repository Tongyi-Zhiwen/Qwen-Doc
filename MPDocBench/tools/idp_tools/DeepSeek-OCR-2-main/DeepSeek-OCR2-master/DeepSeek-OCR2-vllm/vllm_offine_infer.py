from PIL import Image
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from deepseek_ocr2.deepseek_ocr2 import NoRepeatNGramLogitsProcessor
from deepseek_ocr2.config import (
    MAX_CONCURRENCY,
    PROMPT,
)

class DeepseekOCR2Infer:
    max_tokens = 8192
    max_model_len = 8192
    max_num_batched_tokens = 20480
    gpu_memory_utilization = 0.8

    def __init__(self, model_weights_dir: str, lora_weights_dir: str = None):
        self.engine = LLM(
            model=model_weights_dir,
            # hf_overrides={"architectures": ["DeepseekOCR2ForCausalLM"]},
            trust_remote_code=True,
            # block_size=256,
            # enforce_eager=False,
            max_model_len=self.max_model_len,
            # swap_space=0,
            # max_num_seqs=MAX_CONCURRENCY,
            tensor_parallel_size=1,
            gpu_memory_utilization=self.gpu_memory_utilization,
            enable_lora=lora_weights_dir is not None,
        )

        # logits_processors = [NoRepeatNGramLogitsProcessor(
        #     ngram_size=20, window_size=90, whitelist_token_ids={128821, 128822})]

        self.sampling_params = SamplingParams(
            temperature=0.0,
            max_tokens=self.max_tokens,
            # logits_processors=logits_processors,
            skip_special_tokens=False,
            # ignore_eos=False,
        )

        # tokenizer = AutoTokenizer.from_pretrained(model_weights_dir, trust_remote_code=True)
        # self.processor = DeepseekOCR2Processor(tokenizer=tokenizer)

        self.prompt = PROMPT
        self.lora_request = (
            LoRARequest(
                "deepseek_ocr", 1, lora_local_path=lora_weights_dir) if lora_weights_dir is not None else None
        )

    def generate(self, image_file: str):
        image = self.load_image(image_file)
        if image is None:
            return None

        inputs = [
            {
                "prompt": self.prompt,
                "multi_modal_data": {"image": image},
            }
        ]

        response = self.engine.generate(
            inputs, self.sampling_params, lora_request=self.lora_request)
        ocr_result = response[0].outputs[0].text
        tokens_count = len(response[0].outputs[0].token_ids)
        return ocr_result, tokens_count

    def load_image(self, image_file: str):
        try:
            image = Image.open(image_file).convert("RGB")
            return image
        except Exception as e:
            return None


if __name__ == "__main__":
    infer = DeepseekOCR2Infer(model_weights_dir='/nas-mmu/zhoubb/download_models/DeepSeek-OCR-2')
    ocr_result, tokens_count = infer.generate(image_file="/nas-mmu/zhoubb/data/Multi_Page_Documents/Images/book/chn/legal_book_case_0014/page_1.jpg")
    print(ocr_result)
    print(tokens_count)

    ocr_result, tokens_count = infer.generate(image_file="/nas-mmu/zhoubb/data/Multi_Page_Documents/Images/book/chn/legal_book_case_0014/page_1.jpg")
    print(ocr_result)
    print(tokens_count)
