import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from peft import PeftModel, LoraConfig, get_peft_model


def load_base_model(BASE_MODEL_ID="Qwen/Qwen3-VL-2B-Instruct", train=False):
    print(f"Loading base model: {BASE_MODEL_ID}...")
    if train:
        base_model = Qwen3VLForConditionalGeneration.from_pretrained(
            BASE_MODEL_ID,
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
        )
    else:
        try:
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_ID,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                attn_implementation="flash_attention_2",
            )
        except Exception as E:
            print("Flash attention backend not available!")
            base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                BASE_MODEL_ID,
                device_map="auto",
                torch_dtype=torch.float16,
            )
    processor = AutoProcessor.from_pretrained(BASE_MODEL_ID)

    # Training uses right-padding; inference callers override to left-padding as needed
    processor.tokenizer.padding_side = "right"

    return base_model, processor


def load_model_with_lora(LORA_ADAPTER_PATH):
    base_model, processor = load_base_model()
    print(f"Loading LoRA weights from: {LORA_ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(base_model, LORA_ADAPTER_PATH)
    model = model.merge_and_unload()
    model.eval()
    print("Model merged and ready for generation.")

    return model, processor


def lora_config_loader(model_mode="encoder"):

    if model_mode == "encoder":
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    elif model_mode == "encoder-decoder":
        return LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules="all-linear",
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

    raise ValueError("lora config not specified")
