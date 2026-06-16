# Indiana-CXR-Eden
Repository for training a visual language backbone for indian university dataset of report generation.

**A Technical Approach to Medical Imaging, Overcoming Prior Collapse, and Production Scaling**

## 1. Introduction: The Challenge of Medical Imaging

Medical imaging interpretation is inherently complex. It is characterized by high-dimensional visual data, subtle pathological variations, and a severe class imbalance where "normal" findings vastly outnumber true pathologies. 

Vision-Language Models (VLLMs), such as **Qwen3-VL-2B-Instruct**, offer a paradigm shift. By bridging the visual-semantic gap, VLLMs can directly map complex pixel arrays to structured, human-readable clinical narratives. This allows for free-form reasoning and the generation of comprehensive radiology reports. However, adapting a general-domain VLLM to the high-stakes medical domain requires addressing unique engineering, data, and compute challenges.

## 2. Dataset Engineering & Clinical Constraints

To successfully train the VLLM, the raw dataset required significant preprocessing to ensure clinical viability.

- **Clinical Bucketing:** The original dataset contained 121 disparate, often synonymous, and sparse labels (e.g., "Opacity," "Infiltrate," "Airspace Disease"). To combat data sparsity and provide a stable multi-label classification target, these 121 labels were consolidated into **11 clinically distinct buckets** (e.g., *Alveolar Opacities & Infections*, *Pleural Space Diseases*). This reduced noise and resolved synonymous ambiguity.

```jsx
disease_bucket_mapping = {
    # 1. Normal / No Finding
    "normal": "Normal_No_Finding",

    # 2. Cardiomegaly & Cardiovascular Conditions
    "Cardiomegaly": "Cardiomegaly_Cardiovascular",
    "Pulmonary Artery": "Cardiomegaly_Cardiovascular",
    "Cardiac Shadow": "Cardiomegaly_Cardiovascular",
    "Aorta, Thoracic": "Cardiomegaly_Cardiovascular",
    "Aorta": "Cardiomegaly_Cardiovascular",
    "Heart Failure": "Cardiomegaly_Cardiovascular",
    "Epicardial Fat": "Cardiomegaly_Cardiovascular",
    "Heart": "Cardiomegaly_Cardiovascular",
    "Aortic Aneurysm": "Cardiomegaly_Cardiovascular",
    "Heart Atria": "Cardiomegaly_Cardiovascular",
    "Hypertension, Pulmonary": "Cardiomegaly_Cardiovascular",
    "Heart Ventricles": "Cardiomegaly_Cardiovascular",
    "Blood Vessels": "Cardiomegaly_Cardiovascular",
    "Atherosclerosis": "Cardiomegaly_Cardiovascular",
    "Pulmonary Congestion": "Cardiomegaly_Cardiovascular",

    # 3. Alveolar Opacities, Infiltrates & Infections
    "Opacity": "Alveolar_Opacities_Infections",
    "Airspace Disease": "Alveolar_Opacities_Infections",
    "Consolidation": "Alveolar_Opacities_Infections",
    "Infiltrate": "Alveolar_Opacities_Infections",
    "Pulmonary Edema": "Alveolar_Opacities_Infections",
    "Pneumonia": "Alveolar_Opacities_Infections",
    "Bronchitis": "Alveolar_Opacities_Infections",
    "Cavitation": "Alveolar_Opacities_Infections",
    "Bronchiolitis": "Alveolar_Opacities_Infections",

    # 4. Pleural Space Diseases & Pneumothorax
    "Thickening": "Pleural_Space_Pneumothorax",
    "Pleural Effusion": "Pleural_Space_Pneumothorax",
    "Costophrenic Angle": "Pleural_Space_Pneumothorax",
    "Pneumothorax": "Pleural_Space_Pneumothorax",
    "Sulcus": "Pleural_Space_Pneumothorax",
    "Hydropneumothorax": "Pleural_Space_Pneumothorax",
    "Pleura": "Pleural_Space_Pneumothorax",
    "Pericardial Effusion": "Pleural_Space_Pneumothorax",
    "Hemopneumothorax": "Pleural_Space_Pneumothorax",
    "Hemothorax": "Pleural_Space_Pneumothorax",

    # 5. Chronic Obstructive & Hyperinflation Lung Diseases
    "Pulmonary Disease, Chronic Obstructive": "COPD_Hyperinflation",
    "Bullous Emphysema": "COPD_Hyperinflation",
    "Lung, Hyperlucent": "COPD_Hyperinflation",
    "Pulmonary Emphysema": "COPD_Hyperinflation",
    "Emphysema": "COPD_Hyperinflation",
    "Bronchiectasis": "COPD_Hyperinflation",
    "Cystic Fibrosis": "COPD_Hyperinflation",

    # 6. Atelectasis & Volume Loss
    "Pulmonary Atelectasis": "Atelectasis_Volume_Loss",
    "Volume Loss": "Atelectasis_Volume_Loss",
    "Shift": "Atelectasis_Volume_Loss",

    # 7. Nodules, Masses & Chronic Fibrotic/Granulomatous Diseases
    "Pulmonary Fibrosis": "Nodules_Masses_Chronic_Fibrotic",
    "Cicatrix": "Nodules_Masses_Chronic_Fibrotic",
    "Calcified Granuloma": "Nodules_Masses_Chronic_Fibrotic",
    "Granulomatous Disease": "Nodules_Masses_Chronic_Fibrotic",
    "Nodule": "Nodules_Masses_Chronic_Fibrotic",
    "Mass": "Nodules_Masses_Chronic_Fibrotic",
    "Cysts": "Nodules_Masses_Chronic_Fibrotic",
    "Granuloma": "Nodules_Masses_Chronic_Fibrotic",
    "Fibrosis": "Nodules_Masses_Chronic_Fibrotic",
    "Tuberculosis": "Nodules_Masses_Chronic_Fibrotic",
    "Sarcoidosis": "Nodules_Masses_Chronic_Fibrotic",
    "Lymph Nodes": "Nodules_Masses_Chronic_Fibrotic",
    "Lymph nodes": "Nodules_Masses_Chronic_Fibrotic",
    "Lung Diseases, Interstitial": "Nodules_Masses_Chronic_Fibrotic",

    # 8. Spine, Bones & Musculoskeletal Conditions
    "Osteophyte": "Bones_Spine_Musculoskeletal",
    "Spondylosis": "Bones_Spine_Musculoskeletal",
    "Arthritis": "Bones_Spine_Musculoskeletal",
    "Thoracic Vertebrae": "Bones_Spine_Musculoskeletal",
    "Thoracic vertebrae": "Bones_Spine_Musculoskeletal",
    "Bone and Bones": "Bones_Spine_Musculoskeletal",
    "Spine": "Bones_Spine_Musculoskeletal",
    "Scoliosis": "Bones_Spine_Musculoskeletal",
    "Kyphosis": "Bones_Spine_Musculoskeletal",
    "Osteoporosis": "Bones_Spine_Musculoskeletal",
    "Bone Diseases, Metabolic": "Bones_Spine_Musculoskeletal",
    "Dislocations": "Bones_Spine_Musculoskeletal",
    "Shoulder": "Bones_Spine_Musculoskeletal",
    "Spinal Fusion": "Bones_Spine_Musculoskeletal",
    "Expansile Bone Lesions": "Bones_Spine_Musculoskeletal",
    "Fractures, Bone": "Bones_Spine_Musculoskeletal",
    "Lumbar Vertebrae": "Bones_Spine_Musculoskeletal",
    "Sclerosis": "Bones_Spine_Musculoskeletal",
    "Funnel Chest": "Bones_Spine_Musculoskeletal",
    "Cervical Vertebrae": "Bones_Spine_Musculoskeletal",
    "Humerus": "Bones_Spine_Musculoskeletal",
    "Pectus Carinatum": "Bones_Spine_Musculoskeletal",
    "Ribs": "Bones_Spine_Musculoskeletal",
    "Hyperostosis, Diffuse Idiopathic Skeletal": "Bones_Spine_Musculoskeletal",
    "Deformity": "Bones_Spine_Musculoskeletal",

    # 9. Medical Devices & Post-Surgical Support
    "Breast Implants": "Devices_Surgical_Support",
    "Sutures": "Devices_Surgical_Support",
    "Tube, Inserted": "Devices_Surgical_Support",
    "Surgical Instruments": "Devices_Surgical_Support",
    "Stents": "Devices_Surgical_Support",
    "Implanted Medical Device": "Devices_Surgical_Support",
    "Catheters, Indwelling": "Devices_Surgical_Support",
    "Medical Device": "Devices_Surgical_Support",
    "Mastectomy": "Devices_Surgical_Support",
    "Pneumonectomy": "Devices_Surgical_Support",
    "Contrast Media": "Devices_Surgical_Support",

    # 10. Diaphragmatic, Mediastinal & Extrathoracic Alterations
    "Diaphragm": "Diaphragm_Mediastinum_Extrathoracic",
    "Hernia, Hiatal": "Diaphragm_Mediastinum_Extrathoracic",
    "Mediastinum": "Diaphragm_Mediastinum_Extrathoracic",
    "Diaphragmatic Eventration": "Diaphragm_Mediastinum_Extrathoracic",
    "Pneumoperitoneum": "Diaphragm_Mediastinum_Extrathoracic",
    "Trachea": "Diaphragm_Mediastinum_Extrathoracic",
    "Colonic Interposition": "Diaphragm_Mediastinum_Extrathoracic",
    "Cholelithiasis": "Diaphragm_Mediastinum_Extrathoracic",
    "Hernia, Diaphragmatic": "Diaphragm_Mediastinum_Extrathoracic",
    "Trachea, Carina": "Diaphragm_Mediastinum_Extrathoracic",
    "Abdomen": "Diaphragm_Mediastinum_Extrathoracic",
    "Subcutaneous Emphysema": "Diaphragm_Mediastinum_Extrathoracic",
    "Subcutaneous  Emphysema": "Diaphragm_Mediastinum_Extrathoracic",  # Handles the double-space version present in raw data

    # 11. Non-specific Descriptive & Technical Labels
    "Lung": "Non_Specific_Technical",
    "Density": "Non_Specific_Technical",
    "Markings": "Non_Specific_Technical",
    "No Indexing": "Non_Specific_Technical",
    "Calcinosis": "Non_Specific_Technical",
    "Technical Quality of Image Unsatisfactory ": "Non_Specific_Technical",
    "Foreign Bodies": "Non_Specific_Technical",
    "Lucency": "Non_Specific_Technical",
    "Blister": "Non_Specific_Technical",
    "Adipose Tissue": "Non_Specific_Technical",
    "Nipple Shadow": "Non_Specific_Technical",
    "Thorax": "Non_Specific_Technical",
    "Hypovolemia": "Non_Specific_Technical",
}
```

- **The Necessity of Multi-View Inputs:** While many automated systems rely solely on Frontal (PA/AP) views, this pipeline explicitly incorporated **Lateral views**. This is clinically imperative; the lateral view is often the only way to confirm retrocardiac opacities and is highly sensitive for identifying *Pleural Space Diseases* (such as small pleural effusions evidenced by the blunting of the posterior costophrenic sulcus).
- 

![](https://github.com/shaurysrivastav27/Indiana-CXR-Eden/blob/main/utils/pleural_effusion.png)

- **Generalizability & The 200+ Holdout Rationale:** A test set of ~250 cases was strictly partitioned prior to any training manipulations. **This holdout set was specifically chosen via iterative multi-label stratification to maintain the authentic clinical prior distribution** (heavily skewed toward normal findings). Evaluating on an artificially balanced test set yields inflated metrics that do not generalize to a real-world clinical queue. Iterative stratification ensures rare edge-cases exist in the test set without distorting their real-world frequency.

## 3. Compute Strategy: From Free-Tier Colab to Inference

A core objective was selecting a model lightweight enough to train on free-tier compute while still possessing strong emergent reasoning capabilities.

- **Model Selection & LORA:** The 2-Billion parameter Qwen-VL architecture was selected. By applying LORA with bf16 float precision and enabling gradient checkpointing, the VRAM footprint was compressed to comfortably fit within a free-tier 80GB GPU constraint. The inferences were done on L10 GPU with 23GB GPU.
- **Managing VLM Token Explosion:** Passing raw, high-resolution (2000x2000) X-rays into a VLLM results in tens of thousands of tokens, triggering Out-Of-Memory (OOM) errors. Images were systematically resized to 768x768 via Lanczos resampling. This drastic reduction in sequence length unlocked practical batch sizes without degrading clinical quality for macro-pathologies (like Cardiomegaly or Pneumothorax).
- **Inference Efficiency:** During final post-training evaluation on an L10 (using Flash Attention 2 and pure `bfloat16`), the merged checkpoint required **only 7.2 GB of VRAM** with a global batch size of 8, proving its exceptional lightweight profile for downstream deployment.

## 4. Overcoming Prior Collapse: Augmentation & Enhancement

During initial baseline training, the model exhibited severe **Prior Collapse** (Mode Collapse). Because the raw dataset was >80% "Normal/No Finding", the Causal Language Modeling loss incentivized the model to simply predict "normal study" to safely minimize cross-entropy.

- **Why simple text oversampling fails:** In standard NLP, text paraphrasing is used to combat imbalance. In the medical domain, paraphrasing is dangerous because radiologists rely on a highly specific, standardized lexicon. Furthermore, simply duplicating minority rows causes the Vision Encoder to memorize the exact pixel arrangement of a specific patient's scan rather than learning the generalized disease. Although few reports have mention of prior studies and mention details like XXXX. Some sort of cleaning is required for this kind of problems using LLMs.
- **Smart Oversampling + Image Augmentation:** We implemented dynamic oversampling based on the rarest bucket in a given row, duplicating minority classes up to a target threshold. We paired this with **on-the-fly Image Augmentation** (subtle rotations, crops, and contrast shifts). When the model saw the same "Atelectasis" report 15 times, the Vision Encoder saw 15 uniquely augmented representations, forcing true pathological invariance.
- **CLAHE Enhancement:** X-rays are notoriously low-contrast. We introduced Contrast Limited Adaptive Histogram Equalization (CLAHE) preprocessing. CLAHE locally enhances contrast, making ribs, bone lesions, and faint interstitial opacities highly visible for the vision encoder.

## 5. Architectural Evolution: Encoder + Decoder Fine-Tuning

My initial approach relied on fine-tuning only the LLM Decoder via LoRA, leaving the Vision Transformer (ViT) frozen. Pre-trained general ViTs are optimized for internet imagery; they are virtually "blind" to grayscale medical textures.

By expanding my LoRA targets to `all-linear` layers, we explicitly unfroze the projection layers inside the Vision Encoder alongside the Decoder. This allowed the ViT to construct a medically accurate latent space, directly translating radiological textures into embeddings the language model could properly interpret.

## 6. Clinical Evaluation & Ablation Results

Evaluating medical text using standard NLP metrics (like BLEU or ROUGE) is fundamentally flawed, as they rely on exact n-gram matching and fail to capture clinical equivalence. Instead, we prioritized:

1. **CRIMSON Score:** A state-of-the-art metric from the Rajpurkar Lab designed specifically to evaluate factual accuracy and clinical reasoning in radiology, severely penalizing hallucinated conditions or missed critical findings.
2. **RadGraph F1:** A strict, graph-based metric that extracts anatomical entities and checks for exact relational matches.
3. **CheXbert F1:** An automated labeler that extracts clinical presence/absence for key pathologies.

**Ablation Study Results**

| **Setting** | **Image Augment** | **Dataset Size** | **CLAHE** | **BLEU-4** | **RadGraph F1** | **CheXbert-5 Micro F1** | **CRIMSON Score** |GREEN Score|
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **Decoder Only** | No | 3k | No | 13.33 | 0.3395 | 0.0000 | 0.2264 |0.4719618625|
| **Encoder-Decoder** | No | 3k | No | 18.29 | 0.3019 | 0.4444 | 0.2364 |0.3778743423|
| **Encoder-Decoder** | Yes | 7k+ | No | 12.85 | 0.3340 | 0.0900 | 0.4131 |**0.4603635562**|
| **Encoder-Decoder** | **Yes** | **7k+** | **Yes** | **26.44** | **0.3897** | **0.5862** | **0.4616** |0.4320836431|

**Analysis of Results:**

- **The Failure of the Frozen Encoder:** The baseline *Decoder-Only* setting completely collapsed, achieving a CheXbert-5 Micro F1 of **0.000**. The model failed to detect key pathologies and defaulted to "normal," evidenced by a critically low CRIMSON Score of **0.2264**.
- **The Impact of Unfreezing the Encoder:** Unfreezing the vision encoder allowed the model to actually map medical textures, drastically improving the CheXbert-5 score to **0.4444**.
- **The Synergy of Augmentation + CLAHE:** My final setting combining the Encoder/Decoder with Image Augmentation, balanced upsampling (7k+), and **CLAHE** enhancement yielded the best results across every metric. The RadGraph F1 jumped to **0.3897**, and the CRIMSON Score more than doubled from the baseline to **0.4616**. By forcing the model to learn pathological invariance across enhanced images, we successfully mitigated prior collapse.
- Looking at the raw NLP scores they are almost similar for all settings because of most of the cases were normal so on an average the scores are almost same.
- The GREEN score prioritises both normal and abnormal findings thus converging to average which is very close.
- CRIMSON score on the other hand the models abnormal findings and gives preference to that hence there is clear distinction between the models and we are able to establish which model is better.

## 7. Considerations for Scaling in a Real-Life Context

Deploying this VLLM in a live health-tech environment requires transitioning from a training-optimized architecture to a high-throughput, secure inference system.

- **Serving Infrastructure:** The merged LoRA checkpoint (only 7.2 GB) is highly optimal for offline serving engines like **vLLM**. Implementing continuous batching and PagedAttention will allow Eden to handle concurrent API requests from multiple clinics without memory fragmentation or massive hardware costs.
- **Operational Metrics:** Beyond clinical accuracy, production monitoring must heavily weight **Time-to-First-Token (TTFT)** to ensure UI responsiveness for reviewing radiologists, as well as **Cost per 1,000 reports**.
- **Regulatory Compliance & Security:** Real-world scaling requires rigorous DICOM parsing pipelines to strip Protected Health Information (PHI) before the image array hits the GPU to maintain HIPAA compliance. Furthermore, as an assistive diagnostic tool, it enters FDA Software as a Medical Device (SaMD) territory, requiring strict version control, model drift monitoring, and a human-in-the-loop (HITL) fallback mechanism for radiologists to accept/edit generated text.

## 8. Future Scope & Advancements

While the current pipeline demonstrates strong viability, several avenues exist for pushing the system closer to human parity:

- **Post-Training Optimization (DPO):** Standard Cross-Entropy loss treats all token errors equally. In radiology, generating "normal" when a massive pneumothorax is present is a catastrophic error. Implementing Direct Preference Optimization (DPO) using paired (Chosen vs. Rejected) radiologist preferences will teach the model the asymmetric risk of clinical misdiagnoses.
- **Chain-of-Thought (CoT) & The ABCDE Approach:** VLLMs perform better when forced to reason systematically. Future prompts should enforce the traditional radiologist **ABCDE** search pattern (*Airway, Breathing/Lungs, Cardiac, Diaphragm, Extras/Bones*). Forcing the model to explicitly state observations for each anatomical zone before rendering an impression will drastically reduce "satisfaction of search" omissions.
- **Domain-Specific Encoders & Vocabularies:** Future iterations should utilize a natively medical Vision Encoder (e.g., RadDINO) alongside a medically optimized tokenizer. General LLM tokenizers split complex medical terms sub-optimally, whereas a domain-specific vocabulary would increase token efficiency, context window utilization, and semantic comprehension.
