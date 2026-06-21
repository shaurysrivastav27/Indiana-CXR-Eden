# Correction Set — Indiana-CXR-Eden Code Review

---

## Fix 1 — `utils/load_model.py` : `padding_side`
set padding as right during training and left during generate_response function invocation.

---

## Fix 2 — `utils/dataloader.py` : Prompt masking computed from wrong token positions in `QwenVLLMDataCollator`

---

## Fix 3 — `utils/dataloader.py` : CLAHE applied to already-augmented image (earlier version)

CLAHE and augmentation are now independent transforms on the same clean source image.

---

## Fix 4 — `eval/generate_response.py` : used XrayReportDatsset which had assistant prompt during generation. Separate `XRayReportEvalDataset` now which only takes inputs and prompts to model to generate response. 

---

## Fix 7 — `utils/resize_and_save_images.py` : Train and test images saved to the same flat folder with colliding row-index filenames


