import os
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
from transformers import (
    TrainingArguments,
    Trainer,
)
from utils import (
    XRayReportDataset,
    QwenVLLMDataCollator,
    load_base_model,
    lora_config_loader,
)
from peft import get_peft_model


def main(
    model_mode,
    output_dir_root,
    train_csv_path="train_resized.csv",
    test_csv_path="test_resized.csv",
    image_mode="frontal_and_lateral",
):

    output_dir = os.path.join(
        output_dir_root, f"qwen-vl-report-lora-{model_mode}-{image_mode}"
    )

    lora_config = lora_config_loader(model_mode)
    model, processor = load_base_model()

    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    print("Loading Training Dataset...")
    train_dataset = XRayReportDataset(
        csv_file=train_csv_path, image_mode=image_mode, apply_clahe_fn=True
    )

    print("Loading Testing/Validation Dataset...")
    val_dataset = XRayReportDataset(
        csv_file=test_csv_path, image_mode=image_mode, apply_clahe_fn=True
    )

    print(f"Train size: {len(train_dataset)} | Eval size: {len(val_dataset)}")

    collator = QwenVLLMDataCollator(processor)

    # ==========================================
    # 5. Training Arguments
    # ==========================================
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=8,  # Adjust based on VRAM (2B model with 4-bit handles well on 24GB)
        gradient_accumulation_steps=1,
        optim="adamw_torch_fused",  # Fused optimizer for speed
        learning_rate=2e-4,
        fp16=False,
        bf16=True,  # Highly recommended for modern GPUs (Ampere/Hopper)
        max_grad_norm=0.3,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        logging_steps=10,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        dataloader_num_workers=4,
        remove_unused_columns=False,  # Critical: Keep False since we pass raw messages/images
        report_to="tensorboard",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )

    # ==========================================
    # 6. Initialize Trainer & Train
    # ==========================================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collator,
    )

    print("Starting training...")
    trainer.train()

    # Save final model
    trainer.save_model(os.path.join(output_dir, "final_lora"))
    processor.save_pretrained(os.path.join(output_dir, "final_lora"))
    print("Training complete!")


if __name__ == "__main__":
    main()
