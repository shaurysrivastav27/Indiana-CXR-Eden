import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from CRIMSON import CRIMSONScore
# Import Metrics
from bert_score import score as bert_score_fn
from radgraph import F1RadGraph
from green import GREEN  # Stanford AIMI GREEN metric

# ==========================================
# 1. Configuration
# ==========================================
CSV_FILE = "evaluation_results.csv"

# For BERTScore, using a medical-specific BERT model yields much more accurate clinical similarity
BERTSCORE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"

# For GREEN, Stanford provides a fine-tuned 7B model
GREEN_MODEL_ID = "StanfordAIMI/GREEN-radllama2-7b"


def clean_text(text):
    """Handles NaNs and strips whitespace"""
    if pd.isna(text) or str(text).lower() == "nan":
        return ""
    return str(text).strip()


# ==========================================
# Metric Functions
# ==========================================
def compute_crimson(hypotheses, references):
    """
    Computes CRIMSON scores for a list of report pairs.

    Returns:
        crimson_scores: list[float]
        crimson_mean: float
    """

    print("\n[4/4] Computing CRIMSON...")

    scorer = CRIMSONScore()  # Uses MedGemmaCRIMSON by default

    crimson_scores = []

    for ref, hyp in tqdm(
        zip(references, hypotheses),
        total=len(references),
        desc="CRIMSON",
    ):
        try:
            result = scorer.evaluate(
                reference_findings=ref,
                predicted_findings=hyp,
            )

            crimson_scores.append(result["crimson_score"])

        except Exception as e:
            print(f"CRIMSON failed: {e}")
            crimson_scores.append(np.nan)

    crimson_mean = np.nanmean(crimson_scores)

    print(f"-> CRIMSON Score: {crimson_mean:.4f}")

    return crimson_scores, crimson_mean

def compute_bertscore(hypotheses, references):
    """Computes BERTScore using a specified medical BERT model."""
    print("\n[1/3] Computing BERTScore (using PubMedBERT)...")

    # P = Precision, R = Recall, F1 = F1 Score
    P, R, F1_bert = bert_score_fn(
        hypotheses,
        references,
        lang="en",
        model_type=BERTSCORE_MODEL,
        num_layers=12,
        use_fast_tokenizer=True,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    bert_f1_mean = F1_bert.mean().item()
    print(f"-> BERTScore F1: {bert_f1_mean:.4f}")

    # Return the individual scores as a numpy array and the overall mean
    return F1_bert.numpy(), bert_f1_mean


def compute_radgraph(hypotheses, references):
    """Computes RadGraph F1 score."""
    print("\n[2/3] Computing RadGraph...")

    # reward_level="partial" gives credit for overlapping entities/relations
    radgraph_scorer = F1RadGraph(reward_level="partial")

    # RadGraph returns the mean score, list of scores, and the generated graphs
    rg_mean_score, rg_score_list, _, _ = radgraph_scorer(
        hyps=hypotheses, refs=references
    )
    print(f"-> RadGraph F1: {rg_mean_score:.4f}")

    return rg_score_list, rg_mean_score


def compute_green(hypotheses, references):
    """Computes GREEN score using the Stanford AIMI model."""
    print(f"\n[3/3] Computing GREEN (loading {GREEN_MODEL_ID})...")

    # GREEN loads a 7B parameter Llama-2 model internally onto your GPU
    green_scorer = GREEN(
        model_id_or_path=GREEN_MODEL_ID,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    green_mean, green_std, green_score_list = green_scorer(
        refs=references, hyps=hypotheses
    )
    print(f"-> GREEN Score: {green_mean:.4f} ± {green_std:.4f}")

    return green_score_list, green_mean


# ==========================================
# Main Execution
# ==========================================


def main():
    print(f"Loading data from {CSV_FILE}...")
    df = pd.read_csv(CSV_FILE)

    references = []
    hypotheses = []

    # Combine Findings and Impression for full report evaluation
    for _, row in df.iterrows():
        # Ground Truth
        ref_f = clean_text(row.get("findings", ""))
        ref_i = clean_text(row.get("impression", ""))
        ref_full = f"{ref_f} {ref_i}".strip()

        # Model Generated
        hyp_f = clean_text(row.get("generated_findings", ""))
        hyp_i = clean_text(row.get("generated_impression", ""))
        hyp_full = f"{hyp_f} {hyp_i}".strip()

        # Fallback for empty strings to prevent metric crashes
        if not ref_full:
            ref_full = "normal study"
        if not hyp_full:
            hyp_full = "normal study"

        references.append(ref_full)
        hypotheses.append(hyp_full)

    print(f"\nLoaded {len(references)} report pairs for evaluation.")

    # Call the separated metric functions
    bert_scores, bert_f1_mean = compute_bertscore(hypotheses, references)
    rg_scores, rg_mean_score = compute_radgraph(hypotheses, references)
    green_scores, green_mean = compute_green(hypotheses, references)
    crimson_scores, crimson_mean = compute_crimson(hypotheses,references)
    

    # ==========================================
    # Save Results Back to CSV
    # ==========================================
    print("\nSaving granular metrics back to CSV...")
    df["bertscore_f1"] = bert_scores
    df["radgraph_f1"] = rg_scores
    df["green_score"] = green_scores
    df['crimson_score'] = crimson_scores

    output_name = "evaluation_results_with_metrics.csv"
    df.to_csv(output_name, index=False)
    print(f"Done! Final evaluation saved to: {output_name}")

    # Print Final Summary
    print("\n" + "=" * 40)
    print("        FINAL EVALUATION SUMMARY        ")
    print("=" * 40)
    print(f"BERTScore F1 : {bert_f1_mean:.4f}")
    print(f"RadGraph F1  : {rg_mean_score:.4f}")
    print(f"GREEN Score  : {green_mean:.4f}")
    print(f"CRIMSON Score  : {crimson_mean:.4f}")
    print("=" * 40)


if __name__ == "__main__":
    main()
