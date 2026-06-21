import torch
import pandas as pd
import numpy as np
import argparse

BERTSCORE_MODEL = "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract-fulltext"
GREEN_MODEL_ID = "StanfordAIMI/GREEN-radllama2-7b"


def clean_text(text):
    if pd.isna(text) or str(text).lower() == "nan":
        return ""
    return str(text).strip()


def compute_crimson(hypotheses, references):
    from CRIMSON import CRIMSONScore
    from vllm import LLM, SamplingParams

    print("\nComputing CRIMSON...")

    # Use vllm backend for maximum multi-GPU throughput
    scorer = CRIMSONScore(api="vllm")

    # Use batch evaluation instead of sequential loop
    results = scorer.evaluate_batch(
        reference_findings_list=references,
        predicted_findings_list=hypotheses,
        batch_size=8,
    )

    crimson_scores = [r["crimson_score"] if r else np.nan for r in results]
    crimson_mean = np.nanmean(crimson_scores)
    print(f"-> CRIMSON Score: {crimson_mean:.4f}")
    return crimson_scores, crimson_mean


def compute_bertscore(hypotheses, references):
    from transformers import AutoTokenizer, AutoModel

    print("\nComputing BERTScore (PubMedBERT)...")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(BERTSCORE_MODEL, use_fast=False)
    model = AutoModel.from_pretrained(BERTSCORE_MODEL).to(device).eval()

    def embed(texts, batch_size=32):
        all_embs, all_masks = [], []
        for i in range(0, len(texts), batch_size):
            enc = tokenizer(
                texts[i : i + batch_size],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)
            with torch.no_grad():
                emb = model(**enc).last_hidden_state
            emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            all_embs.append(emb)
            all_masks.append(enc["attention_mask"])
        return all_embs, all_masks

    hyp_embs, hyp_masks = embed(hypotheses)
    ref_embs, ref_masks = embed(references)

    F1_scores = []
    idx = 0
    for batch_h, batch_hm, batch_r, batch_rm in zip(
        hyp_embs, hyp_masks, ref_embs, ref_masks
    ):
        for j in range(batch_h.size(0)):
            h = batch_h[j][batch_hm[j].bool()]
            r = batch_r[j][batch_rm[j].bool()]
            sim = h @ r.T
            P = sim.max(dim=1).values.mean()
            R = sim.max(dim=0).values.mean()
            F1 = 2 * P * R / (P + R + 1e-8)
            F1_scores.append(F1.item())

    F1_arr = np.array(F1_scores)
    bert_f1_mean = F1_arr.mean()
    print(f"-> BERTScore F1: {bert_f1_mean:.4f}")
    return F1_arr, bert_f1_mean


def compute_radgraph(hypotheses, references):
    from radgraph import F1RadGraph

    print("\nComputing RadGraph...")
    radgraph_scorer = F1RadGraph(reward_level="partial")
    rg_mean_score, rg_score_list, _, _ = radgraph_scorer(
        hyps=hypotheses, refs=references
    )
    print(f"-> RadGraph F1: {rg_mean_score:.4f}")
    return rg_score_list, rg_mean_score


def compute_green(hypotheses, references):
    import sys
    import os

    sys.path.append(os.path.expanduser("~/SageMaker/efs/eshaurys/CH/GREEN/"))
    from green_score.green import GREEN

    print(f"\nComputing GREEN ({GREEN_MODEL_ID})...")
    green_scorer = GREEN(
        model_id_or_path=GREEN_MODEL_ID,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    green_mean, green_std, green_score_list = green_scorer(
        refs=references, hyps=hypotheses
    )
    print(f"-> GREEN Score: {green_mean:.4f} ± {green_std:.4f}")
    return green_score_list, green_mean


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate radiology report generation metrics"
    )
    parser.add_argument(
        "--input_csv", required=True, help="Input CSV with generated reports"
    )
    parser.add_argument(
        "--output_csv", required=True, help="Output CSV with metric scores"
    )
    parser.add_argument("--bertscore", action="store_true", help="Compute BERTScore")
    parser.add_argument("--radgraph", action="store_true", help="Compute RadGraph F1")
    parser.add_argument("--green", action="store_true", help="Compute GREEN score")
    parser.add_argument("--crimson", action="store_true", help="Compute CRIMSON score")
    args = parser.parse_args()

    print(f"Loading data from {args.input_csv}...")
    df = pd.read_csv(args.input_csv)

    references, hypotheses = [], []
    for _, row in df.iterrows():
        ref_full = f"{clean_text(row.get('findings', ''))} {clean_text(row.get('impression', ''))}".strip()
        hyp_full = f"{clean_text(row.get('generated_findings', ''))} {clean_text(row.get('generated_impression', ''))}".strip()
        references.append(ref_full or "normal study")
        hypotheses.append(hyp_full or "normal study")

    print(f"Loaded {len(references)} report pairs.")

    summary = {}

    if args.bertscore:
        scores, mean = compute_bertscore(hypotheses, references)
        df["bertscore_f1"] = scores
        summary["BERTScore F1"] = mean

    if args.radgraph:
        scores, mean = compute_radgraph(hypotheses, references)
        df["radgraph_f1"] = scores
        summary["RadGraph F1"] = mean

    if args.green:
        scores, mean = compute_green(hypotheses, references)
        df["green_score"] = scores
        summary["GREEN Score"] = mean

    if args.crimson:
        scores, mean = compute_crimson(hypotheses, references)
        df["crimson_score"] = scores
        summary["CRIMSON Score"] = mean

    df.to_csv(args.output_csv, index=False)
    print(f"\nSaved to: {args.output_csv}")

    if summary:
        print("\n" + "=" * 40)
        print("        EVALUATION SUMMARY        ")
        print("=" * 40)
        for name, val in summary.items():
            print(f"{name:<15}: {val:.4f}")
        print("=" * 40)


if __name__ == "__main__":
    main()
