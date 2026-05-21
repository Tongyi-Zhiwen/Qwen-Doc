def register():
    from vllm import ModelRegistry

    if "DeepseekOCR2ForCausalLM" not in ModelRegistry.get_supported_archs():
        from .deepseek_ocr2 import DeepseekOCR2ForCausalLM

        ModelRegistry.register_model("DeepseekOCR2ForCausalLM", DeepseekOCR2ForCausalLM)