import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.v2 as T
import cv2
import numpy as np

system_prompt = """Perform a systematic review of this chest X-ray against the following clinical categories:
1. Cardiomegaly & Cardiovascular
2. Alveolar Opacities & Infections
3. Pleural Space Diseases
4. COPD & Hyperinflation
5. Atelectasis & Volume Loss
6. Nodules, Masses & Fibrosis
7. Bones & Spine
8. Medical Devices
9. Diaphragm & Extrathoracic

Based on this systematic review, generate the findings and impression in the xml format as specified below.

output format:

<findings> Findings </findings>
<impression> Impression </impression>
"""


def generate_input_prompt_template(indication):
    return f"Given the indication:\n\nINDICATION : {indication}\n\n{system_prompt}"


def generate_output_prompt_template(findings, impression):
    return f"<findings> {findings} </findings>\n<impression> {impression} </impression>"


def preprocess_dataset(dataset_csv_path):
    df = pd.read_csv(dataset_csv_path)

    # Fill missing text fields with empty strings
    df["indication"] = df["indication"].fillna("None available.")
    df["findings"] = df["findings"].fillna("None available.")
    df["impression"] = df["impression"].fillna("None available.")

    return df


def apply_clahe(pil_image):
    # Convert PIL to CV2 grayscale
    img_cv = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl1 = clahe.apply(img_cv)
    # Convert back to RGB PIL
    return Image.fromarray(cv2.cvtColor(cl1, cv2.COLOR_GRAY2RGB))


class XRayReportDataset(Dataset):
    def __init__(
        self,
        csv_file,
        is_train=True,
        image_mode="frontal_and_lateral",
        apply_clahe_fn=False,
    ):
        """
        Args:
            csv_file (str): Path to the dataset CSV.
            image_mode (str): "frontal_only" or "frontal_and_lateral"
        """
        self.df = preprocess_dataset(csv_file)
        self.image_mode = image_mode
        self.augment = None
        self.is_train = is_train
        self.apply_clahe_fn = apply_clahe_fn

        if self.is_train:
            self.augment = T.Compose(
                [
                    # Very slight rotations (patients breathe and tilt slightly)
                    T.RandomRotation(degrees=(-5, 5)),
                    # Slight zooming/cropping (simulates different distance from X-ray machine)
                    T.RandomResizedCrop(
                        size=(768, 768), scale=(0.95, 1.0), ratio=(0.95, 1.05)
                    ),
                    # Subtle brightness/contrast shifts (simulates different X-ray exposure settings)
                    T.ColorJitter(brightness=0.1, contrast=0.1),
                ]
            )

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Prepare Text Inputs and Targets
        indication = str(row["indication"]).strip()
        findings = str(row["findings"]).strip()
        impression = str(row["impression"]).strip()

        prompt_text = generate_input_prompt_template(indication)
        target_text = generate_output_prompt_template(findings, impression)

        # 2. Gather Images
        images = []
        content_list = []

        frontal_img_path = row["Frontal"]
        lateral_img_path = row["Lateral"]

        # Load Frontal Image
        if isinstance(frontal_img_path, str) and os.path.exists(frontal_img_path):
            try:
                front_img = Image.open(frontal_img_path).convert("RGB")
                if self.apply_clahe_fn:
                    # apply clahe for bones related disorders
                    clahe_front_img = apply_clahe(front_img)
                    images.append(clahe_front_img)
                    content_list.append({"type": "image", "image": clahe_front_img})
                if self.augment:
                    front_img = self.augment(front_img)
                
                
                images.append(front_img)
                content_list.append({"type": "image", "image": front_img})


            except Exception as e:
                print(f"Error loading frontal image {row['Frontal']}: {e}")

        # Load Lateral Image (if required and available)
        if (
            self.image_mode == "frontal_and_lateral"
            and isinstance(lateral_img_path, str)
            and os.path.exists(lateral_img_path)
        ):
            try:
                lat_img = Image.open(lateral_img_path).convert("RGB")
                if self.augment:
                    lat_img = self.augment(lat_img)
                images.append(lat_img)
                content_list.append({"type": "image", "image": lat_img})
            except Exception as e:
                print(f"Error loading lateral image {row['Lateral']}: {e}")

        # 3. Append the text instruction to the content list
        content_list.append({"type": "text", "text": prompt_text})

        # 4. Construct the Qwen-VL Chat Message Format
        messages = [
            {"role": "user", "content": content_list},
            {"role": "assistant", "content": [{"type": "text", "text": target_text}]},
        ]

        return {"messages": messages, "images": images}


class XRayReportEvalDataset(Dataset):
    def __init__(
        self,
        csv_file,
        image_mode="frontal_and_lateral",
        apply_clahe_fn=False,
    ):
        """
        Args:
            csv_file (str): Path to the dataset CSV.
            image_mode (str): "frontal_only" or "frontal_and_lateral"
        """
        self.df = preprocess_dataset(csv_file)
        self.image_mode = image_mode
        self.apply_clahe_fn = apply_clahe_fn

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # 1. Prepare Text Inputs and Targets
        indication = str(row["indication"]).strip()
        findings = str(row["findings"]).strip()
        impression = str(row["impression"]).strip()

        prompt_text = generate_input_prompt_template(indication)
        target_text = generate_output_prompt_template(findings, impression)

        # 2. Gather Images
        images = []
        content_list = []

        frontal_img_path = row["Frontal"]
        lateral_img_path = row["Lateral"]

        # Load Frontal Image
        if isinstance(frontal_img_path, str) and os.path.exists(frontal_img_path):
            try:
                front_img = Image.open(frontal_img_path).convert("RGB")

                if self.apply_clahe_fn:
                    # apply clahe for bones related disorders
                    clahe_front_img = apply_clahe(front_img)
                    images.append(clahe_front_img)
                    content_list.append({"type": "image", "image": clahe_front_img})

                images.append(front_img)
                content_list.append({"type": "image", "image": front_img})

                

            except Exception as e:
                print(f"Error loading frontal image {row['Frontal']}: {e}")

        # Load Lateral Image (if required and available)
        if (
            self.image_mode == "frontal_and_lateral"
            and isinstance(lateral_img_path, str)
            and os.path.exists(lateral_img_path)
        ):
            try:
                lat_img = Image.open(lateral_img_path).convert("RGB")
                images.append(lat_img)
                content_list.append({"type": "image", "image": lat_img})
            except Exception as e:
                print(f"Error loading lateral image {row['Lateral']}: {e}")

        # 3. Append the text instruction to the content list
        content_list.append({"type": "text", "text": prompt_text})

        # 4. Construct the Qwen-VL Chat Message Format
        messages = [
            {"role": "user", "content": content_list}
        ]

        return {"messages": messages, "images": images}

class QwenVLLMDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        texts = []
        image_inputs = []

        # To mask the prompt, we need the raw prompt text lengths
        prompt_texts = []

        for feature in features:
            messages = feature["messages"]
            images = feature["images"]

            # Format the full sequence (User + Assistant)
            full_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            texts.append(full_text)

            # Format the prompt only (User only) to calculate masking boundary
            prompt_only = self.processor.apply_chat_template(
                [messages[0]], tokenize=False, add_generation_prompt=True
            )
            prompt_texts.append(prompt_only)

            # Qwen processor expects a flat list of all images in the batch
            image_inputs.extend(images)

        # Process the full sequences (Inputs + Targets)
        batch = self.processor(
            text=texts,
            images=image_inputs if len(image_inputs) > 0 else None,
            return_tensors="pt",
            padding=True,
        )

        labels = batch["input_ids"].clone()

        # Masking out the user prompt and padding tokens
        for i in range(len(labels)):
            # 1. Find where real (non-padding) tokens start in the full sequence.
            #    With left-padding, padding tokens sit at the front, so the first
            #    position where attention_mask == 1 is where the prompt begins.
            real_start = (batch["attention_mask"][i] == 1).nonzero(as_tuple=True)[0][0].item()

            # 2. Get the unpadded prompt length by tokenizing the prompt alone,
            #    without any batch padding, so we get the true token count.
            prompt_encoding = self.processor.tokenizer(
                prompt_texts[i], return_tensors="pt", padding=False
            )
            prompt_len = prompt_encoding["input_ids"].shape[1]

            # 3. Mask exactly the prompt tokens in the full sequence.
            labels[i, real_start : real_start + prompt_len] = -100

            # 4. Mask padding tokens.
            pad_mask = batch["attention_mask"][i] == 0
            labels[i, pad_mask] = -100

        batch["labels"] = labels
        return batch
