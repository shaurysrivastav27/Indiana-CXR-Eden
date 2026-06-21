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

![](https://github.com/shaurysrivastav27/Indiana-CXR-Eden/blob/main/utils/pleural_effusion.png)

- **Generalizability & The 200+ Holdout Rationale:** A test set of 198 cases was strictly partitioned prior to any training manipulations. **This holdout set was specifically chosen via iterative multi-label stratification to maintain the authentic clinical prior distribution** (heavily skewed toward normal findings). Evaluating on an artificially balanced test set yields inflated metrics that do not generalize to a real-world clinical queue. Iterative stratification ensures rare edge-cases exist in the test set without distorting their real-world frequency.

## 3. Compute Strategy: From Free-Tier Colab to Inference

A core objective was selecting a model lightweight enough to train on free-tier compute while still possessing strong emergent reasoning capabilities.

- **Model Selection & LORA:** The 2-Billion parameter Qwen-VL architecture was selected. By applying LORA with bf16 float precision and enabling gradient checkpointing, the VRAM footprint was compressed to comfortably fit within a free-tier GPU constraint. The inferences and trainings both were done on L10 GPU with 23GB GPU.
- **Managing VLM Token Explosion:** Passing raw, high-resolution (2000x2000) X-rays into a VLLM results in tens of thousands of tokens, triggering Out-Of-Memory (OOM) errors. Images were systematically resized to 768x768 via Lanczos resampling. This drastic reduction in sequence length unlocked practical batch sizes without degrading clinical quality for macro-pathologies (like Cardiomegaly or Pneumothorax).
- **Inference Efficiency:** During final post-training evaluation on an L10 (using Flash Attention 2 and pure `bfloat16`), the merged checkpoint required **only 7.2 GB of VRAM** with a global batch size of 8, proving its exceptional lightweight profile for downstream deployment.

## 4. Overcoming Prior Collapse: Augmentation & Enhancement

During initial baseline training, the model exhibited severe **Prior Collapse** (Mode Collapse). Because the raw dataset was >80% "Normal/No Finding", the Causal Language Modeling loss incentivized the model to simply predict "normal study" to safely minimize cross-entropy. Hence the metrics for No-finding and cases which had more prevelance were giving good metrics, specifically Pleural effusion (in settings where laterals were present). 

- **Why simple text oversampling fails:** In standard NLP, text paraphrasing is used to combat imbalance. In the medical domain, paraphrasing is dangerous because radiologists rely on a highly specific, standardized lexicon. Furthermore, simply duplicating minority rows causes the Vision Encoder to memorize the exact pixel arrangement of a specific patient's scan rather than learning the generalized disease. Although few reports have mention of prior studies and mention details like XXXX. Some sort of cleaning is required for this kind of problems using LLMs.
- **Smart Oversampling + Image Augmentation:** We implemented dynamic oversampling based on the rarest bucket in a given row, duplicating minority classes up to a target threshold. We paired this with **on-the-fly Image Augmentation** (subtle rotations, crops, and contrast shifts). When the model saw the same "Atelectasis" report 15 times, the Vision Encoder saw 15 uniquely augmented representations, forcing true pathological invariance.
- **CLAHE Enhancement:** X-rays are notoriously low-contrast. We introduced Contrast Limited Adaptive Histogram Equalization (CLAHE) preprocessing. CLAHE locally enhances contrast, making ribs, bone lesions, and faint interstitial opacities highly visible for the vision encoder.

## 5. Architectural Evolution: Encoder + Decoder Fine-Tuning

My initial approach relied on fine-tuning only the LLM Decoder via LoRA, leaving the Vision Transformer (ViT) frozen. Pre-trained general ViTs are optimized for internet imagery; they are virtually "blind" to grayscale medical textures.

By expanding my LoRA targets to `all-linear` layers, I explicitly unfroze the projection layers inside the Vision Encoder alongside the Decoder. This allowed the ViT to construct a medically accurate latent space, directly translating radiological textures into embeddings the language model could properly interpret.

## 6. Clinical Evaluation & Ablation Results

Evaluating medical text using standard NLP metrics (like BLEU or ROUGE) is fundamentally flawed, as they rely on exact n-gram matching and fail to capture clinical equivalence. Instead, we prioritized:

1. **CRIMSON Score:** A state-of-the-art metric from the Rajpurkar Lab designed specifically to evaluate factual accuracy and clinical reasoning in radiology, severely penalizing hallucinated conditions or missed critical findings.
2. **RadGraph F1:** A strict, graph-based metric that extracts anatomical entities and checks for exact relational matches.
3. **CheXbert F1:** An automated labeler that extracts clinical presence/absence for key pathologies.

**Ablation Study Results**

All metrics below are measured on the held-out stratified test set. BLEU and RougeL are included for completeness but are not the primary evaluation signal (see analysis below). BERTScore uses PubMedBERT embeddings. CRIMSON and RadGraph F1 are the primary clinical accuracy metrics.

**Baseline (no fine-tuning)**

| **View** | **BLEU** | **RougeL** | **RadGraph F1** | **CheXbert Accuracy** | **CheXbert Micro F1** | **BERTScore** | **CRIMSON** |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frontal only | 0.80 | 0.1957 | 0.1575 | 0.0202 | 0.1237 | 0.9002 | 0.3028 |
| Frontal + Lateral | 0.84 | 0.1979 | 0.1585 | 0.0051 | 0.1240 | 0.9004 | 0.2100 |

**Encoder-Decoder LoRA (`all-linear` targets)**

| **Views** | **Oversample** | **CLAHE** | **BLEU** | **RougeL** | **RadGraph F1** | **CheXbert Accuracy** | **CheXbert Micro F1** | **BERTScore** | **CRIMSON** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frontal only | No | No | 9.58 | 0.3648 | 0.3006 | 0.6313 | 0.2069 | 0.9412 | 0.2751 |
| Frontal + Lateral | No | No | 10.52 | 0.3782 | 0.3214 | 0.7172 | 0.3488 | 0.9421 | 0.3315 |
| Frontal + Lateral | Yes | No | 13.70 | 0.3800 | 0.3104 | 0.6465 | 0.2549 | 0.9414 | 0.0552 |
| Frontal + Lateral | Yes | Yes | 9.31 | 0.3502 | 0.2742 | 0.5707 | 0.2331 | 0.9379 | -0.1034 |

**Encoder LoRA (selective `q/k/v/o/gate/up/down` targets)**

| **Views** | **CLAHE** | **BLEU** | **RougeL** | **RadGraph F1** | **CheXbert Accuracy** | **CheXbert Micro F1** | **BERTScore** | **CRIMSON** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Frontal only | No | 9.41 | 0.3744 | 0.2952 | 0.6465 | 0.2586 | 0.9411 | 0.1492 |
| Frontal only | Yes | 9.83 | 0.3688 | 0.2982 | 0.7323 | 0.1987 | 0.9420 | 0.2777 |
| Frontal + Lateral | No | 9.61 | 0.3759 | 0.3110 | 0.7071 | 0.3444 | 0.9419 | 0.2850 |
| **Frontal + Lateral** | **Yes** | **10.58** | **0.3825** | **0.3048** | **0.7121** | **0.3343** | **0.9420** | **0.3857** |

**Analysis of Results:**

- **Baseline collapse is confirmed.** The untuned model achieves near-zero CheXbert accuracy (0.5–2%) on both view configurations. It defaults to predicting "No Finding" for nearly every sample, as shown by the extremely low CheXbert Micro F1 (~0.12). RadGraph F1 of ~0.16 reflects shallow surface-level overlap only.

- **Encoder-Decoder with no oversampling is the strongest Encoder-Decoder setting.** `Frontal + Lateral, No CLAHE, No Oversample` reaches the highest RadGraph F1 (0.3214) and CheXbert accuracy (0.7172) in that group, as well as the highest CRIMSON (0.3315). Adding oversampling without careful augmentation appears to hurt CRIMSON significantly — the oversample+CLAHE variant produces a negative CRIMSON score (-0.1034), indicating the model is generating hallucinated or clinically incorrect findings. This likely points to the oversampling introducing repetitive samples that the model memorizes rather than generalizing from, compounded by the CLAHE channel adding ambiguity when the model hasn't fully learned to separate the two image streams.

- **Encoder-only LoRA with Frontal + Lateral + CLAHE is the best overall setting.** It achieves the highest CRIMSON score (0.3857), highest RougeL (0.3825), competitive RadGraph F1 (0.3048), and the best CheXbert accuracy (0.7121) among all fine-tuned runs. The selective LoRA targeting (q/k/v/o/gate/up/down) provides more stable gradients than `all-linear`, avoiding the instability seen in the Encoder-Decoder runs under oversampling.

- **CLAHE consistently helps the Encoder-only setting.** Comparing Frontal+Lateral Encoder runs: without CLAHE CRIMSON is 0.2850, with CLAHE it rises to 0.3857 — a ~35% relative improvement. The CLAHE channel provides the encoder with an enhanced structural view that improves bone and interstitial finding detection without disrupting the standard view's pathological signal.

- **NLP metrics (BLEU, RougeL) are poor discriminators here.** All fine-tuned settings cluster in a narrow BLEU range (9–14) and RougeL range (0.35–0.38) regardless of clinical quality. This confirms that standard text overlap metrics are insensitive to clinical correctness and should not be used as the primary signal for model selection in radiology report generation.

- **BERTScore is uniformly high (~0.94) across all fine-tuned runs** and provides almost no discriminative signal between settings. The baseline (~0.90) is meaningfully lower, confirming fine-tuning helps semantic similarity, but BERTScore alone cannot distinguish between clinically accurate and clinically incorrect fine-tuned outputs.

- **CRIMSON is the most discriminative metric.** It shows the largest variance across settings (ranging from -0.1034 to 0.3857) and correctly identifies the expected failure modes — the oversample+CLAHE Encoder-Decoder variant scores negatively, which aligns with expected hallucination behavior under that training regime.

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


---

## 9. Qualitative Analysis of Generated Reports

This section examines generated radiology reports across all ten model configurations by comparing them against ground-truth findings and impressions sampled from the Indiana University Chest X-Ray dataset. Samples were drawn to cover normal cases, pathology-positive cases (emphysema, pleural effusion, cardiomegaly, atelectasis, consolidation, rib fracture), and edge cases with very short or very complex ground truth. The analysis is organised by observed pattern.

---

### 9.1 Baseline Output Quality: XML-Tagged Template vs. Free-Text Generation

The two baseline configurations (`results_baseline_frontal.csv`, `results_baseline_frontal_lateral.csv`) use a fundamentally different generation paradigm from the eight improved models. Their outputs are verbose, XML-structured responses organised around anatomical categories (`<cardiomegaly_and_cardiovascular>`, `<alveolar_opacities_and_infections>`, etc.), whereas every other configuration generates compact free-text findings and impressions in the style of a real radiology report.

This structural difference has direct clinical consequences.

**Normal case (uid 46) — baseline_frontal:**
> *Ground truth findings:* "None available. Well-expanded and clear lungs. Mediastinal contour within normal limits. No acute cardiopulmonary abnormality identified."
> *Generated findings:* `<alveolar_opacities_and_infections> The lung fields appear clear with no significant alveolar opacities or signs of infection such as consolidation, nodules, or infiltrates.</alveolar_opacities_and_infections>` [... repeated across 8 anatomical XML tags ...]

**Same case — frontal_lateral_no_clahe_no_oversample_encoder_decoder:**
> *Generated findings:* "The heart is normal in size. The mediastinum is unremarkable. There is no pleural effusion, pneumothorax, or focal airspace disease. Mild degenerative changes are present within the spine."
> *Generated impression:* "No acute cardiopulmonary abnormality."

The baseline output is many times longer, fills all sections with boilerplate negatives, and is not directly usable in a clinical workflow. The improved models produce concise, clinically idiomatic text that closely mirrors the ground-truth style. This explains the dramatic metric gap: the baselines score ROUGE-L ≈ 0.196, BLEU ≈ 0.8, and CheXbert ≈ 0.02, compared to ROUGE-L ≈ 0.37–0.38, BLEU ≈ 9–14, and CheXbert ≈ 0.63–0.73 for the improved models. BERTScore is the one metric where the gap narrows (0.900 baseline vs. ~0.941 improved), because cosine similarity over sentence embeddings is less sensitive to verbosity and structural mismatch than n-gram or label-based metrics.

---

### 9.2 Pathology Recall: What the Models Miss and Misattribute

#### 9.2.1 Emphysema and Hyperinflation

For the emphysema case (uid 25), ground truth explicitly describes "lungs are hyperlucent and hyperinflated compatible with emphysema," "moderate left pleural effusion and small right pleural effusion," and "left lower lobe airspace disease." This is one of the most pathologically complex cases in the shared test set.

No configuration correctly identifies all three pathology domains. The responses across configurations fall into three clusters:

- **Encoder-only models (frontal+lateral)** tend to detect one salient sign and drop the rest. `frontal_lateral_no_clahe_no_oversample_encoder` generates: *"There are large bilateral pleural effusions with associated atelectasis. No pneumothorax."* — correctly flagging effusion but missing emphysema and lateralising incorrectly ("bilateral" vs. "left greater than right").

- **Encoder-decoder models** generate plausible but often hallucinated content. `frontal_lateral_no_clahe_no_oversample_encoder_decoder` (uid 25) produces: *"There is an 8 cm right upper quadrant mass which may represent a tumor. No pneumothorax."* — a high-confidence fabrication of a mass with no basis in the ground truth.

- **CLAHE + oversample encoder-decoder** generates: *"There are calcified hilar lymph nodes bilaterally. There is no focal airspace disease. No pleural effusion or pneumothorax."* — a false-negative sweep that misses all positive findings.

**Pattern:** Emphysema and concurrent effusion are reliably missed unless one sign dominates the image. No model captures the full constellation.

#### 9.2.2 Cardiomegaly

Cardiomegaly is the most consistently detected pathology across all non-baseline configurations. For uid 437 (ground truth: "There is cardiomegaly. The contour of the ascending aorta is prominent, consistent with known ascending aortic aneurysm"), the encoder-decoder models without oversampling correctly identify the enlarged heart:

**frontal_lateral_no_clahe_no_oversample_encoder_decoder (uid 437):**
> *Generated findings:* "The heart is enlarged. There are postsurgical changes of prior CABG with mediastinal clips. No focal consolidation or pleural effusion."
> *Generated impression:* "1. Cardiomegaly without pulmonary edema. 2. Postsurgical changes of prior CABG."

This is genuinely good output — it identifies cardiomegaly, CABG changes, and absence of edema, matching the clinical substance even if missing the aortic aneurysm. By contrast, `frontal_no_clahe_no_oversample_encoder` for the same case generates: *"Enlarged cardiac silhouette with mild pulmonary vascular congestion. This may be secondary to cardiomegaly or pulmonary edema."* — hallucinating pulmonary vascular congestion not in the ground truth.

**Pattern:** Cardiomegaly is the most consistently recalled pathology (CheXbert Cardiomegaly F1 peaks at 0.387 for frontal_lateral_no_clahe_no_oversample_encoder vs. near-zero for baselines), but several configurations still substitute hallucinated secondary signs (congestion, edema) when none are present.

#### 9.2.3 Subtle Findings: Atelectasis, Streaky Opacities, Hiatal Hernia

For cases where the ground truth describes a subtle finding, the improved models frequently generate a plausible-but-wrong finding or a correct general impression with a wrong finding:

**uid 357** — Ground truth: "streaky right medial basilar airspace opacities, possibly due to airspace disease or atelectasis. The right heart border appears obscured."
- `frontal_lateral_no_clahe_no_oversample_encoder_decoder` generates: *"The heart is normal size. There is no focal air space opacity to suggest a pneumonia."* — misses both the obscured border and the basilar opacities entirely.
- `frontal_lateral_clahe_no_oversample_encoder` generates: *"There is no pleural effusion, pneumothorax, or focal airspace disease."* — a clean false negative.
- `frontal_lateral_clahe_oversample_encoder_decoder` generates: *"The heart size is mildly enlarged. Mild vascular prominence of the left lung."* — invents cardiomegaly on a normal-sized heart, directly contradicting ground truth.

**uid 296** (hiatal hernia) — Ground truth: "Heart size normal. Lungs clear. Hiatal hernia." Neither encoder nor encoder-decoder model in any configuration mentions the hiatal hernia. This is consistent with the low CheXbert Diaphragm/Mediastinum coverage across all runs.

**uid 500** (cardiomegaly + nerve stimulator device) — `frontal_lateral_no_clahe_no_oversample_encoder_decoder` reports: *"The heart is normal size. There are calcified mediastinal lymph nodes. The lungs are clear without infiltrate."* — misses both the cardiomegaly and the device. `frontal_lateral_clahe_oversample_encoder_decoder` generates findings = "None available" with an impression fabricated from a blank input, producing: *"Heart size is normal. Mild streaky atelectasis left base."* — a complete hallucination.

---

### 9.3 Hallucination Patterns

Hallucinations fall into three observable types across the dataset:

**Type 1 — Calcified granuloma over-generation.** CLAHE-enabled frontal-only encoder models (`frontal_clahe_no_oversample_encoder`, `frontal_no_clahe_no_oversample_encoder`) systematically insert "scattered calcified granulomas" into otherwise normal reports. For uid 171 (ground truth: "Well-expanded and clear lungs. No acute cardiopulmonary abnormality identified."), `frontal_no_clahe_no_oversample_encoder` generates: *"There are scattered calcified granulomas within the lungs. Lungs are otherwise clear."* No granulomas are mentioned in the ground truth. This pattern appears in at least 8 of 28 sampled rows for frontal-only encoder models, suggesting the model has learned "calcified granulomas" as a safe filler phrase that appears commonly in the training distribution.

**Type 2 — Emphysema false positives on normal lungs.** The `frontal_lateral_no_clahe_oversample_encoder_decoder` configuration repeatedly generates emphysematous changes for clear-lung cases. For uid 66 (ground truth: "Both lungs are clear and expanded with no pleural air collections"), the generated output is: *"Mild emphysematous changes without focal consolidation."* Impression: *"Emphysema without acute disease."* This exact phrase appears again for uid 171, uid 200, and uid 363, all ground-truth normal cases — indicating mode collapse toward a single safe-abnormality template.

**Type 3 — Fabricated masses and devices.** `frontal_lateral_no_clahe_no_oversample_encoder_decoder` generates a wholly fabricated pathology for uid 25: *"There is an 8 cm right upper quadrant mass which may represent a tumor."* This is the most clinically dangerous hallucination type and appears concentrated in the no-CLAHE no-oversample encoder-decoder models, possibly because they lack regularisation from contrast enhancement.

---

### 9.4 Effect of CLAHE Preprocessing

Comparing CLAHE vs. no-CLAHE configurations within the same architecture reveals a consistent but nuanced effect:

| Configuration | ROUGE-L | RadGraph | CRIMSON |
|---|---|---|---|
| frontal_no_clahe_no_oversample_encoder | 0.374 | 0.295 | 0.149 |
| frontal_clahe_no_oversample_encoder | 0.369 | 0.298 | 0.278 |
| frontal_lateral_no_clahe_no_oversample_encoder | 0.376 | 0.311 | 0.285 |
| frontal_lateral_clahe_no_oversample_encoder | 0.383 | 0.305 | 0.386 |

CLAHE provides a small boost to CRIMSON scores (which reward clinical coherence) without consistently improving RadGraph or ROUGE-L. Qualitatively, this manifests as slightly more organised reasoning rather than more accurate pathology detection.

**CLAHE (frontal+lateral encoder) — uid 437:**
> *"The heart is mildly enlarged but stable. There are tortuosity and ectasia of the thoracic aorta. No focal consolidation, pneumothorax or large pleural effusion. Mild degenerative changes of the spine."*
> *Impression: "1. Stable mild cardiomegaly without acute abnormality."*

**No-CLAHE (frontal+lateral encoder) — uid 437:**
> *"The heart is enlarged. There are tortuosity of the thoracic aorta. No focal airspace disease or effusion. No pneumothorax. No acute bony abnormality."*
> *Impression: "1. Enlarged heart with tortuous thoracic aorta. 2. Negative for acute cardiopulmonary process."*

Both correctly detect cardiomegaly and aortic tortuosity. The CLAHE version adds "mildly" and "stable" — qualifiers that are clinically accurate (this is a known aneurysm on follow-up) but not in the ground truth either. CLAHE does not introduce new hallucinations here; it mainly refines hedging language.

The negative side: CLAHE increases the false-positive rate for granuloma mentions in frontal-only models (see §9.3 Type 1), likely because enhanced contrast makes benign calcifications more visually salient during training.

---

### 9.5 Effect of Oversampling

Oversampling rare pathologies is intended to improve recall for underrepresented conditions (effusion, atelectasis, edema). In practice the results are mixed:

**`frontal_lateral_no_clahe_oversample_encoder_decoder`** achieves the highest BLEU score (13.7) and competitive ROUGE-L (0.380), but its CRIMSON score collapses to 0.055 — the worst of all ten configurations. Qualitative inspection reveals why: the model adopts a repetitive template that recycles phrasing regardless of image content.

For uid 66 (clear normal lungs):
> *"The heart is normal in size. The mediastinum is unremarkable. Mild emphysematous changes without focal consolidation. There is no pleural effusion or pneumothorax."*
> *Impression: "Emphysema without acute disease."*

This exact phrasing appears verbatim for uid 171, uid 200, uid 363 — four clean-lung patients all receive an emphysema diagnosis. Oversampling biased the model toward producing pathology outputs even on normal cases, causing false-positive inflation that BLEU/ROUGE reward (because "emphysema" is a real clinical word with correct sentence structure) but CRIMSON and CheXbert penalise (because the clinical entity is wrong).

A contributing factor: the `is_normal` heuristic used to classify cases during oversampling was initially only excluding two conditions (`cardiomegaly` and `opacity`). Cases with pleural effusion, atelectasis, pneumothorax, emphysema, and fractures were misclassified as "normal" and excluded from oversampling, meaning the "abnormal" set being upsampled was itself contaminated and incomplete. This was corrected post-evaluation (see Corrections).

**`frontal_lateral_clahe_oversample_encoder_decoder`** fares worse still: CRIMSON = -0.103, the only negative CRIMSON score in the set. The model hallucinates devices and vascular findings on clean scans and produces "None available" as generated findings for uid 22 and uid 500. This suggests CLAHE + oversampling together destabilise the encoder-decoder model, possibly by amplifying contrast artifacts that mimic abnormal features during training.

---

### 9.6 Multi-View (Frontal + Lateral) vs. Frontal Only

Adding the lateral view consistently improves RadGraph scores and CheXbert accuracy across paired comparisons:

| Comparison | RadGraph |
|---|---|
| frontal_no_clahe_no_oversample_encoder | 0.295 |
| frontal_lateral_no_clahe_no_oversample_encoder | 0.311 |
| frontal_no_clahe_no_oversample_encoder_decoder | 0.301 |
| frontal_lateral_no_clahe_no_oversample_encoder_decoder | 0.321 |

Qualitatively, the lateral view enables the model to pick up findings that are visible only in the lateral projection. For uid 185 (ground truth: "Lungs are mildly hypoinflated. Increased opacities on lateral projection reflect bronchovascular crowding"):

**Frontal-only encoder:** *"The heart is normal in size. The mediastinum is unremarkable. The lungs are clear."* — misses hypoinflation entirely.

**Frontal+lateral encoder:** *"The heart is normal in size. The mediastinum is unremarkable. There are scattered calcified granulomas within the lungs. Lungs are otherwise clear."* — still misses hypoinflation, but the finding mention is distinct from the frontal-only output, indicating the lateral view influenced the generation even if inaccurately.

**Frontal+lateral encoder-decoder (no-CLAHE, no oversample):** *"The heart is normal in size. The mediastinum is unremarkable. Mild blunting of the right costophrenic angle compatible with small pleural effusion or scarring. There is no pneumothorax."* — a mild false positive (no effusion in ground truth) but demonstrates that the lateral view prompts the model to reason about costophrenic angles, which are only fully visible on the lateral projection.

The lateral view appears most beneficial for multi-compartment cases. For straightforward normal cases, adding a second view primarily adds noise by giving the model more surface area on which to hallucinate.

---

### 9.7 Encoder vs. Encoder-Decoder Architecture

The encoder-only architecture (classification head generating text directly from image features) and the encoder-decoder architecture (full sequence-to-sequence generation) show a clear qualitative divergence:

**Encoder models** generate short, factual reports. They are less prone to hallucination but more prone to incomplete reports — frequently stopping after "No acute disease" without describing anything about the actual image. For a case like uid 437 (cardiomegaly + aortic aneurysm), `frontal_lateral_no_clahe_no_oversample_encoder` produces a two-sentence output correctly naming cardiomegaly but ignoring the ascending aorta. This brevity scores lower on ROUGE and BLEU (fewer words = less n-gram overlap) but higher on precision-oriented metrics.

**Encoder-decoder models** generate richer, longer descriptions. `frontal_lateral_no_clahe_no_oversample_encoder_decoder` for uid 437:
> *"The heart is enlarged. There are postsurgical changes of prior CABG with mediastinal clips. No focal consolidation or pleural effusion. Calcified right hilar lymph nodes. Degenerative changes thoracic spine."*
> *Impression: "1. Cardiomegaly without pulmonary edema. 2. Postsurgical changes of prior CABG."*

This report mentions four distinct observations correctly. However, encoder-decoder models are also the source of the most egregious hallucinations (§9.3 Type 2 and 3). The tradeoff is recall versus precision: encoder-decoder models surface more findings but also generate more false positives.

For the frontal-only configurations without preprocessing benefits, the encoder-decoder introduces the most variance. `frontal_no_clahe_no_oversample_encoder_decoder` produces `findings = "None available"` for uid 343 (which has real consolidation and pulmonary congestion), and its impression for this blank finding is: *"Heart size is normal. Lungs are clear. No edema or effusions."* — a complete miss generated from no visual evidence.

---

### 9.8 Why Metrics Sometimes Disagree

The observed output differences explain several metric divergences that are not intuitive from numbers alone:

**BLEU/ROUGE vs. CheXbert.** `frontal_lateral_no_clahe_oversample_encoder_decoder` achieves BLEU = 13.7 (best of all ten) but CheXbert accuracy = 0.646 and CRIMSON = 0.055 (among the worst). The reason is that its mode-collapsed "emphysema without acute disease" template is lexically close to normal-case ground truths ("no acute disease"), boosting n-gram overlap, while simultaneously assigning the wrong pathology label — CheXbert detects this mismatch. ROUGE and BLEU reward surface fluency; CheXbert rewards clinical correctness.

**RadGraph vs. BERTScore.** BERTScore is remarkably flat across configurations (0.900–0.942) because it operates over contextualised embeddings of full sentences, and all models produce grammatical, domain-appropriate medical text. RadGraph, by contrast, extracts clinical entity-relation triples ("pleural effusion / present / bilateral") and checks them against ground truth — much more sensitive to whether the right pathology is named at all. For baseline models, BERTScore = 0.900 but RadGraph = 0.157, because the XML-structured output uses correct medical vocabulary but is structurally divorced from the ground-truth format RadGraph entity extraction was trained on.

**CRIMSON vs. all others.** CRIMSON rewards report-level clinical coherence (internally consistent findings, appropriate hedging, no logical contradictions). The `frontal_lateral_clahe_oversample_encoder_decoder` model is the only one to achieve a negative CRIMSON score (-0.103), driven by reports that simultaneously state "None available" for findings while giving a specific impression, or that describe normal-size hearts in findings but mention cardiomegaly in impressions. RadGraph and BLEU do not detect this internal inconsistency.

---

### 9.9 Overall Assessment: Which Configuration Produces the Most Clinically Useful Reports?

> **Note on evaluation validity:** The ablation results in section 6 and the qualitative comparisons above were produced before a critical evaluation bug was identified and fixed: `generate_response.py` was using the training dataset class, which included the ground-truth assistant turn (findings + impression) in the input prompt. The model was completing a partially pre-filled sequence rather than generating blindly. All metrics are therefore optimistic across all fine-tuned configurations.
**`frontal_lateral_no_clahe_no_oversample_encoder_decoder`** produces the most consistently useful reports overall. It has:
- The highest RadGraph score of all ten configurations (0.321)
- Strong CheXbert accuracy (0.717) — second best
- Competitive ROUGE-L (0.378) and BLEU (10.52)
- CRIMSON = 0.332 — best among the encoder-decoder configurations
- Reports that use proper clinical language, identify cardiomegaly and postsurgical changes correctly, and hedge appropriately ("may represent atelectasis or scarring")

Its reports are also the right length — typically 3–5 sentences of findings and 1–2 sentence impressions — matching real radiology reports.

The `frontal_lateral_clahe_no_oversample_encoder` gives a competitive alternative (CRIMSON = 0.386, highest of all), but is limited by shorter, sometimes incomplete reports that miss secondary findings. For use cases where CRIMSON coherence is the priority, it wins; where pathology recall is paramount, the encoder-decoder variant is stronger.

The configurations to avoid clinically are the two oversampled encoder-decoder models, particularly `frontal_lateral_clahe_oversample_encoder_decoder`, whose negative CRIMSON score and blank-findings pattern indicate an unstable model that would be unreliable in practice.

---
