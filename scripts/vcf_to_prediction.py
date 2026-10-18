#!/usr/bin/env python3
"""
vcf_to_prediction.py
====================
End-to-end workflow: VCF file → resistance prediction + SHAP values,
annotated with the WHO mutation catalogue (second edition).

Usage:
    python3 scripts/vcf_to_prediction.py \
        --vcf test_data/ERR040120.filtered.vcf.gz \
        --drug RIFAMPICIN \
        --model-dir models/ \
        --output-dir results/predictions/

     # All 4 drugs at once
    python3 scripts/vcf_to_prediction.py \
        --vcf test_data/ERR040120.filtered.vcf.gz \
        --all-drugs

    # Predict, then interpret with the agent (backend/model from config)
    python3 scripts/vcf_to_prediction.py \
        --vcf test_data/ERR040120.filtered.vcf.gz \
        --all-drugs \
        --interpret \
        --config scripts/agent/agent_config.yaml
Output:
    results/predictions/ERR040120_predictions.json
    results/predictions/ERR040120_RIFAMPICIN_shap_values.csv

Each drug record includes an explanation that leads with the SHAP
contributions and, where the isolate's allele matches the catalogue,
the WHO variant name and final confidence grading. Matching uses
NC_000962.3 position, reference nucleotide, and alternate nucleotide
against scripts/WHO-UCN-TB-2023.7-eng.xlsx.

For the same prediction plus a per-sample waterfall plot and a
cohort beeswarm plot, use scripts/vcf_to_shap_plots.py.

Trained models:
    HuggingFace: https://huggingface.co/nanzhen102/FORUM-TB-models
    Download models to models/ before running.    
"""

import os
import sys
import gzip
import json
import argparse
from pathlib import Path

import pandas as pd
import joblib
import shap

from who_catalogue import (
    SOURCE as WHO_SOURCE,
    annotate_feature,
    build_explanation,
    iter_alleles,
    load_catalogue,
    who_table_fields,
)

# ============================================================
# SETTINGS
# ============================================================
DRUGS = ["RIFAMPICIN", "ISONIAZID", "ETHAMBUTOL", "PYRAZINAMIDE"]

# Second-edition WHO mutation catalogue, placed next to this script.
DEFAULT_CATALOGUE = (
    Path(__file__).resolve().parent / "WHO-UCN-TB-2023.7-eng.xlsx"
)

# Nucleotide encoding (same as ml_matrix.csv.gz)
NUC_ENCODE = {"A": 1, "T": 2, "C": 3, "G": 4}

# AMR gene coordinates on H37Rv (NC_000962.3)
AMR_GENES = {
    "rpoB":  (759807,  763325),
    "katG":  (2153889, 2156111),
    "inhA":  (1674202, 1675011),
    "fabG1": (1673440, 1674183),
    "embB":  (4246514, 4249810),
    "embA":  (4243233, 4246517),
    "embC":  (4239863, 4243147),
    "pncA":  (2288681, 2289241),
    "rpsA":  (1833542, 1834987),
}
# ============================================================


class VariantCalls(dict):
    """SNP calls keyed by position, plus every allele for catalogue matching.

    SNP values are {"ref", "alt"}. ``alleles`` also includes indels and
    multi-nucleotide alleles as (pos, ref, alt) tuples so they can be matched
    to the WHO catalogue even though the model scores SNPs only.
    """

    def __init__(self):
        super().__init__()
        self.alleles = []


def _alt_base(call):
    """ALT string from a SNP call. Accepts {"ref", "alt"} or a bare ALT."""
    if isinstance(call, dict):
        return call.get("alt", "")
    return call


def parse_vcf(vcf_path):
    """
    Parse a VCF or VCF.gz file.

    SNP calls (single REF base, single ALT base) are stored as
    {position: {"ref", "alt"}} and are what the model encodes.
    Every ACGT allele, including indels, is kept on ``variants.alleles``
    for exact WHO catalogue matching.
    """
    variants = VariantCalls()
    seen = set()
    opener = gzip.open if str(vcf_path).endswith(".gz") else open

    with opener(vcf_path, "rt") as f:
        for line in f:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 5:
                continue
            pos = int(parts[1])
            ref = parts[3].upper()
            alts = [allele.upper() for allele in parts[4].split(",") if allele and allele != "."]

            for alt in alts:
                if not alt or any(base not in "ACGT" for base in ref + alt):
                    continue
                key = (pos, ref, alt)
                if key not in seen:
                    seen.add(key)
                    variants.alleles.append(key)
                # The model has one allele per position. Keep the historical
                # rule: a SNP line whose ALT field is a single base.
                if len(ref) == 1 and len(alts) == 1 and len(alt) == 1:
                    variants[pos] = {"ref": ref, "alt": alt}

    print(
        f"  Parsed {len(variants)} SNPs from VCF "
        f"({len(variants.alleles)} alleles for catalogue matching)"
    )
    return variants


def encode_sample(variants, feature_columns):
    """
    Encode a sample's variants as a feature vector
    matching the ml_matrix.csv.gz format.
    Missing positions → 0 (assume REF).
    """
    # Start with all zeros (REF)
    feature_vector = pd.Series(0, index=feature_columns, dtype=int)

    matched = 0
    for pos, call in variants.items():
        col = f"pos_{pos}"
        if col in feature_vector.index:
            feature_vector[col] = NUC_ENCODE.get(_alt_base(call), 0)
            matched += 1

    print(f"  {matched} variants matched to AMR gene positions")
    return feature_vector


def get_gene_for_position(pos):
    """Return gene name for a genomic position, or None."""
    pos = int(pos.replace("pos_", ""))
    for gene, (start, end) in AMR_GENES.items():
        if start <= pos <= end:
            return gene
    return "unknown"


def nonzero_shap(shap_series):
    """Keep features whose SHAP value is not exactly 0.0."""
    return shap_series[shap_series != 0]


def predict_and_explain(sample_id, feature_vector, drug, model_dir,
                        variants=None, catalogue=None):
    """
    Load trained RF model, predict resistance, compute SHAP values,
    and attach WHO catalogue grades where the isolate's allele matches.
    """
    model_path = os.path.join(model_dir, f"rf_{drug}_v2.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    print(f"  Loading model: {model_path}")
    rf = joblib.load(model_path)

    # Reshape to 2D array (1 sample)
    X = pd.DataFrame([feature_vector], columns=feature_vector.index)

    # Prediction
    prob = rf.predict_proba(X)[0][1]   # probability of Resistant
    label = "Resistant" if prob >= 0.5 else "Susceptible"

    print(f"  Prediction: {label} (probability: {prob:.4f})")

    # SHAP values — interventional mode with background for numerical stability
    print("  Computing SHAP values...")
    df_bg = pd.read_csv(
        "resistance_dataset/ml_matrix.csv.gz",
        index_col="SAMPLE",
        usecols=["SAMPLE"] + list(feature_vector.index)
    ).dropna().sample(100, random_state=42)

    explainer = shap.TreeExplainer(
        rf,
        data=df_bg,
        feature_perturbation="interventional"
    )
    shap_vals = explainer.shap_values(X, check_additivity=False)

    # Handle both old (list) and new (3-D array) SHAP output formats
    if isinstance(shap_vals, list):
        shap_array = shap_vals[1][0]
    else:
        shap_array = shap_vals[0, :, 1] if shap_vals.ndim == 3 else shap_vals[0]

    shap_series = nonzero_shap(pd.Series(shap_array, index=feature_vector.index))
    # Top 20 features by absolute SHAP value (zeros already dropped)
    top_shap = shap_series.abs().nlargest(20)
    top_shap_details = [
        annotate_feature(
            feat,
            get_gene_for_position(feat),
            int(feature_vector[feat]),
            float(shap_series[feat]),
            variants,
            catalogue,
            drug,
        )
        for feat in top_shap.index
    ]

    known_variants = []
    if catalogue is not None:
        encoded_by_pos = {}
        shap_by_pos = {}
        for feat in feature_vector.index:
            pos = int(str(feat).replace("pos_", ""))
            encoded_by_pos[pos] = int(feature_vector[feat])
            # nonzero_shap() drops exactly-zero features, so a missing name
            # means a SHAP of 0.0 rather than an error.
            shap_by_pos[pos] = float(shap_series.get(feat, 0.0))
        known_variants = catalogue.known_resistance_variants(
            iter_alleles(variants), drug, shap_by_pos, encoded_by_pos
        )

    explanation = build_explanation(
        sample_id,
        drug,
        label,
        round(float(prob), 4),
        top_shap_details,
        known_variants,
        catalogue_loaded=catalogue is not None,
    )

    result = {
        "sample":     sample_id,
        "drug":       drug,
        "prediction": label,
        "probability_resistant": round(float(prob), 4),
        "explanation": explanation,
        "who_catalogue_source": WHO_SOURCE if catalogue is not None else None,
        "known_resistance_variants": known_variants,
        "top_shap_features": top_shap_details,
    }

    return result, shap_series


def main():
    parser = argparse.ArgumentParser(
        description="VCF → resistance prediction + SHAP values"
    )
    parser.add_argument("--vcf",        required=True,
                        help="Path to VCF or VCF.gz file")
    parser.add_argument("--drug",       default="RIFAMPICIN",
                        choices=DRUGS,
                        help="Drug to predict resistance for")
    parser.add_argument("--model-dir",  default="models/",
                        help="Directory containing trained .joblib models")
    parser.add_argument("--output-dir", default="results/predictions/",
                        help="Directory to save output files")
    parser.add_argument("--all-drugs",  action="store_true",
                        help="Run prediction for all 4 drugs")
    parser.add_argument("--catalogue", default=str(DEFAULT_CATALOGUE),
                        help="WHO mutation-catalogue workbook (.xlsx)")
    parser.add_argument("--interpret", action="store_true",
                        help="After prediction, run the SHAP interpretation agent")
    parser.add_argument("--config",
                        default=str(
                            Path(__file__).resolve().parent
                            / "agent" / "agent_config.yaml"
                        ),
                        help="Agent config YAML/JSON (backend and model). "
                             "Used with --interpret.")
    parser.add_argument("--backend", choices=["anthropic", "claude", "openai"],
                        help="Override the agent config backend")
    parser.add_argument("--llm-model", dest="llm_model",
                        help="Override the agent config model id")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Get sample ID from filename
    sample_id = os.path.basename(args.vcf)
    for suffix in (".filtered.vcf.gz", ".vcf.gz", ".vcf"):
        if sample_id.endswith(suffix):
            sample_id = sample_id[: -len(suffix)]
            break
    print(f"\n{'='*60}")
    print(f"Sample: {sample_id}")
    print(f"VCF:    {args.vcf}")
    print(f"{'='*60}")

    # Load feature columns from ml_matrix
    print("\nLoading feature columns from ml_matrix...")
    matrix_path = "resistance_dataset/ml_matrix.csv.gz"
    df_cols = pd.read_csv(matrix_path, nrows=0, index_col="SAMPLE")
    feature_cols = [c for c in df_cols.columns if c.startswith("pos_")]
    print(f"  {len(feature_cols)} AMR gene positions loaded")

    # Parse VCF
    print(f"\nParsing VCF: {args.vcf}")
    variants = parse_vcf(args.vcf)

    # Encode sample
    print("\nEncoding sample...")
    feature_vector = encode_sample(variants, feature_cols)

    catalogue = None
    catalogue_path = Path(args.catalogue)
    if catalogue_path.is_file():
        catalogue = load_catalogue(str(catalogue_path))
    else:
        print(f"\nWHO catalogue not found ({catalogue_path}).")
        print("Explanations will use SHAP values only.")

    # Predict for one or all drugs
    drugs_to_run = DRUGS if args.all_drugs else [args.drug]

    all_results = {}
    for drug in drugs_to_run:
        print(f"\n--- {drug} ---")
        try:
            result, shap_series = predict_and_explain(
                sample_id, feature_vector, drug, args.model_dir,
                variants=variants, catalogue=catalogue,
            )
            all_results[drug] = result

            # Save SHAP values CSV
            shap_path = os.path.join(
                args.output_dir,
                f"{sample_id}_{drug}_shap_values.csv"
            )
            encoded = feature_vector.reindex(shap_series.index)
            who_rows = [
                who_table_fields(annotate_feature(
                    feat,
                    get_gene_for_position(feat),
                    int(value),
                    float(shap_value),
                    variants,
                    catalogue,
                    drug,
                ))
                for feat, value, shap_value in zip(
                    shap_series.index, encoded.values, shap_series.values
                )
            ]
            shap_df = pd.DataFrame({
                "position":      shap_series.index,
                "gene":          [get_gene_for_position(p)
                                  for p in shap_series.index],
                "encoded_value": encoded.values,
                "shap_value":    shap_series.values,
                "who_variant":   [row["who_variant"] for row in who_rows],
                "who_mutation":  [row["who_mutation"] for row in who_rows],
                "who_effect":    [row["who_effect"] for row in who_rows],
                "who_grading":   [row["who_grading"] for row in who_rows],
            })
            shap_df = shap_df.sort_values(
                "shap_value", key=abs, ascending=False
            )
            shap_df.to_csv(shap_path, index=False)
            print(f"  SHAP values saved: {shap_path}")

        except FileNotFoundError as e:
            print(f"  SKIPPED: {e}")

    # Save prediction JSON
    json_path = os.path.join(
        args.output_dir,
        f"{sample_id}_predictions.json"
    )
    with open(json_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nPredictions saved: {json_path}")

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    for drug, res in all_results.items():
        emoji = "🔴" if res["prediction"] == "Resistant" else "🟢"
        print(f"  {emoji} {drug:<15}: {res['prediction']:<12} "
              f"(prob: {res['probability_resistant']:.4f})")
    print(f"{'='*60}")
    print("EXPLANATIONS")
    print(f"{'='*60}")
    for drug, res in all_results.items():
        print(f"\n{drug}\n")
        print(res.get("explanation", ""))
    print(f"\n{'='*60}\n")

    if args.interpret:
        _run_interpretation(json_path, args)


def _run_interpretation(json_path, args):
    """Load the agent from scripts/agent/ and interpret the saved JSON."""
    agent_dir = Path(__file__).resolve().parent / "agent"
    sys.path.insert(0, str(agent_dir))
    try:
        import shap_agent
    except ImportError as exc:
        raise SystemExit(
            "Could not import the interpretation agent from scripts/agent/.\n"
            f"{exc}"
        ) from exc

    try:
        cfg = shap_agent.resolve_agent_config(
            config_path=args.config,
            backend=args.backend,
            model=args.llm_model,
        )
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Agent config error: {exc}") from exc

    if not shap_agent.credentials_available(backend=cfg.backend,
                                            config_path=args.config):
        print(shap_agent.credential_help(cfg))
        print("Predictions were saved; interpretation was skipped.")
        return

    print(f"Running interpretation agent ({cfg.backend} / {cfg.model})...")
    report = shap_agent.interpret(json_path, config=cfg)
    print(report)
    report_path = Path(json_path).with_name(
        Path(json_path).name.replace("_predictions.json", "_interpretation.md")
    )
    report_path.write_text(report)
    print(f"\nInterpretation saved: {report_path}")


if __name__ == "__main__":
    main()