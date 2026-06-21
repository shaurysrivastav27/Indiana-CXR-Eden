import os
import torch
import pandas as pd
from PIL import Image
from tqdm import tqdm
from utils import (
    generate_input_prompt_template,
    generate_output_prompt_template,
    preprocess_dataset,
    load_base_model,
    load_model_with_lora,
    XRayReportEvalDataset,
)
import argparse


def parse_images(message):
    content = message[0]["content"]
    images = []
    for item in content:
        if item["type"] == "image":
            images.append(item["image"])
    return images


def generate_reports(
    LORA_ADAPTER_PATH=None,
    TEST_CSV="test_resized.csv",
    OUTPUT_CSV="evaluation_results.csv",
    MAX_NEW_TOKENS=1024,
    IMAGE_MODE="frontal_and_lateral",
    BATCH_SIZE=8,
    apply_clahe_fn=False,
):

    # ==========================================
    # 4. Prepare Dataset
    # ==========================================
    print(f"Loading test dataset: {TEST_CSV}")
    df = XRayReportEvalDataset(
        csv_file=TEST_CSV, image_mode=IMAGE_MODE, apply_clahe_fn=apply_clahe_fn
    )

    generated_reports = []

    model, processor = load_model_with_lora(LORA_ADAPTER_PATH)
    processor.tokenizer.padding_side = "left"

    total_len = len(df)

    # ==========================================
    # 5. Batched Generation Loop
    # ==========================================
    for i in tqdm(
        range(0, total_len, BATCH_SIZE), desc=f"Generating Reports (BS={BATCH_SIZE})"
    ):

        batch_df = [df[idx] for idx in range(i, min(BATCH_SIZE + i, total_len))]

        texts = []
        batch_images = []

        for row in batch_df:
            # Apply chat template for this specific row
            messages = row["messages"]
            batch_images.extend(parse_images(messages))
            text_prompt = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            texts.append(text_prompt)

        # Tokenize the entire batch at once
        inputs = processor(
            text=texts,
            images=batch_images if len(batch_images) > 0 else None,
            return_tensors="pt",
            padding=True,
        ).to("cuda")

        # Generate!
        with torch.no_grad():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                repetition_penalty=1.1,
            )

        # Slice the generated_ids to remove the prompt tokens
        generated_ids_trimmed = [
            out_ids[len(in_ids) :]
            for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]

        # Decode the entire batch back to text
        output_texts = processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )

        for out_text in output_texts:
            generated_reports.append(out_text.strip())

    # ==========================================
    # 6. Save Results
    # ==========================================
    df = df.df
    df["generated_report"] = generated_reports

    def extract_tag(text, tag):
        try:
            start = text.split(f"<{tag}>")[1]
            return start.split(f"</{tag}>")[0].strip()
        except IndexError:
            return ""

    df["generated_findings"] = df["generated_report"].apply(
        lambda x: extract_tag(x, "findings")
    )
    df["generated_impression"] = df["generated_report"].apply(
        lambda x: extract_tag(x, "impression")
    )

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nFinished! Results saved to {OUTPUT_CSV}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Qwen-VL with LoRA for X-Ray reports"
    )
    parser.add_argument("--lora_path", default=None, help="Train CSV path")
    parser.add_argument(
        "--test_csv", default="test_resized.csv", help="Test/val CSV path"
    )
    parser.add_argument(
        "--output_csv",
        default="test_output_resized.csv",
        help="Test/val output CSV path",
    )
    parser.add_argument(
        "--max_new_tokens",
        default=1024,
    )
    parser.add_argument(
        "--image_mode",
        default="frontal_and_lateral",
        choices=["frontal_and_lateral", "frontal", "lateral"],
    )
    parser.add_argument("--batch_size", default=8, type=int)
    parser.add_argument("--apply_clahe_fn", action="store_true")
    parser.add_argument(
        "--tensor_parallel_size",
        default=1,
        type=int,
        help="VLLM tensor parallel size (number of GPUs)",
    )

    args = parser.parse_args()

    os.environ["VLLM_TENSOR_PARALLEL_SIZE"] = str(args.tensor_parallel_size)

    generate_reports(
        LORA_ADAPTER_PATH=args.lora_path,
        TEST_CSV=args.test_csv,
        OUTPUT_CSV=args.output_csv,
        MAX_NEW_TOKENS=int(args.max_new_tokens),
        IMAGE_MODE=args.image_mode,
        BATCH_SIZE=args.batch_size,
        apply_clahe_fn=args.apply_clahe_fn,
    )
