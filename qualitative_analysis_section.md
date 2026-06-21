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

**`frontal_lateral_no_clahe_no_oversample_encoder_decoder`** produces the most consistently useful reports overall. It has:
- The highest RadGraph score of all ten configurations (0.321)
- Strong CheXbert accuracy (0.717) — second best
- Competitive ROUGE-L (0.378) and BLEU (10.52)
- CRIMSON = 0.332 — best among the encoder-decoder configurations
- Reports that use proper clinical language, identify cardiomegaly and postsurgical changes correctly, and hedge appropriately ("may represent atelectasis or scarring")

Its reports are also the right length — typically 3–5 sentences of findings and 1–2 sentence impressions — matching real radiology reports.

The `frontal_lateral_clahe_no_oversample_encoder` gives a competitive alternative (CRIMSON = 0.386, highest of all), but is limited by shorter, sometimes incomplete reports that miss secondary findings. For use cases where CRIMSON coherence is the priority, it wins; where pathology recall is paramount, the encoder-decoder variant is stronger.

The configurations to avoid clinically are the two oversampled encoder-decoder models, particularly `frontal_lateral_clahe_oversample_encoder_decoder`, whose negative CRIMSON score and blank-findings pattern indicate an unstable model that would be unreliable in practice.
