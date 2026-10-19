#!/usr/bin/env python3
"""
shap_agent.py
=============
A tool-using agent that interprets FORUM-TB SHAP attributions.

Consumes the prediction JSON produced by `vcf_to_prediction.py` and turns the
raw attribution table into a written interpretation, looking up the supporting
evidence (amino-acid change, cohort prevalence, co-resistance context) rather
than recalling it.

Backend and model come from `scripts/agent/agent_config.yaml` (anthropic or
openai). Override with `--config`, `--backend`, or `--model`.

Usage:
    # Full report on all four drugs
    python3 scripts/agent/shap_agent.py \\
        --predictions results/predictions/ERR040120_predictions.json

    # A specific question, OpenAI backend from the config or CLI
    python3 scripts/agent/shap_agent.py \\
        --predictions results/predictions/ERR040120_predictions.json \\
        --backend openai --model gpt-4o \\
        --question "Why is this isolate predicted pyrazinamide-resistant?"

    # Show which tools the agent called (verify claims against evidence)
    python3 scripts/agent/shap_agent.py --predictions ... --trace

Anthropic needs ANTHROPIC_API_KEY (or `ant auth login`). OpenAI needs
OPENAI_API_KEY. The four tools below are plain functions over local data and
need no credentials — they can be imported and tested offline.
"""

import os
import sys
import json
import inspect
import argparse
import contextvars
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import pandas as pd

_SCRIPTS = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_SCRIPTS))
try:
    # Single source of truth for the AMR gene coordinates.
    from vcf_to_prediction import AMR_GENES
except ImportError as exc:  # pragma: no cover - dependency guard
    raise ImportError(
        "Could not import AMR_GENES from vcf_to_prediction.py. That module "
        "requires numpy/pandas/joblib/shap — install the project dependencies "
        "(pip install -r dashboard/requirements.txt)."
    ) from exc

from shap_agent_prompts import SYSTEM_PROMPT, REPORT_REQUEST, QUESTION_REQUEST

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG = Path(__file__).resolve().parent / "agent_config.yaml"
GENOME_FASTA = REPO_ROOT / "reference" / "H37Rv.fasta"
ML_MATRIX = REPO_ROOT / "resistance_dataset" / "ml_matrix.csv.gz"
CROSS_DRUG_DIR = REPO_ROOT / "results"

DRUGS = ["RIFAMPICIN", "ISONIAZID", "ETHAMBUTOL", "PYRAZINAMIDE"]
BACKENDS = ("anthropic", "openai")

# Strand for each AMR gene on H37Rv (NC_000962.3). Validated at import time by
# _validate_reading_frames(): every gene must have a length divisible by 3, a
# terminal stop codon, and no internal stop on the strand recorded here.
GENE_STRAND = {
    "rpoB": "+", "katG": "-", "inhA": "+", "fabG1": "+", "embB": "+",
    "embA": "+", "embC": "+", "pncA": "-", "rpsA": "+",
}

# Independently established resistance mutations, used to verify that the
# codon arithmetic reproduces known biology before any residue is reported.
FRAME_ANCHORS = [
    (761155, "rpoB", 450, "S"),
    (2155168, "katG", 315, "S"),
    (4247429, "embB", 306, "M"),
]

NUC_DECODE = {0: "REF", 1: "A", 2: "T", 3: "C", 4: "G"}
COMPLEMENT = {"A": "T", "T": "A", "C": "G", "G": "C", "N": "N"}

_BASES = "TCAG"
_AAS = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON_TABLE = {
    b1 + b2 + b3: _AAS[i]
    for i, (b1, b2, b3) in enumerate(
        (x, y, z) for x in _BASES for y in _BASES for z in _BASES
    )
}

# Set per-call by interpret(); a ContextVar rather than a module global so
# concurrent Streamlit sessions don't clobber each other.
_ACTIVE = contextvars.ContextVar("active_predictions", default=None)


# ============================================================
# Config
# ============================================================

@dataclass
class AgentConfig:
    backend: str = "anthropic"
    model: str = "claude-opus-5"
    effort: str = "high"
    max_tokens: int = 16000

    def __post_init__(self):
        backend = str(self.backend or "anthropic").strip().lower()
        if backend == "claude":
            backend = "anthropic"
        if backend not in BACKENDS:
            raise ValueError(
                f"Unknown backend {self.backend!r}. Expected one of: "
                f"anthropic, claude, openai."
            )
        self.backend = backend
        self.model = str(self.model or "").strip()
        if not self.model:
            raise ValueError("Config 'model' must be a non-empty model id.")
        self.effort = str(self.effort or "high").strip().lower()
        self.max_tokens = int(self.max_tokens)


def _coerce_config_value(raw):
    if isinstance(raw, (int, float, bool)) or raw is None:
        return raw
    text = str(raw).strip().strip('"').strip("'")
    lowered = text.lower()
    if lowered in ("true", "false"):
        return lowered == "true"
    try:
        return int(text)
    except ValueError:
        return text


def _parse_config_text(text: str) -> dict:
    """Accept JSON or a flat YAML/INI-style `key: value` file."""
    stripped = text.strip()
    if not stripped:
        return {}
    if stripped[0] in "{[":
        data = json.loads(stripped)
        if not isinstance(data, dict):
            raise ValueError("Config JSON must be an object.")
        return data

    data = {}
    for line in stripped.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        if key:
            data[key] = _coerce_config_value(value)
    return data


def load_agent_config(path=None) -> AgentConfig:
    """Load `scripts/agent/agent_config.yaml`, or `path` if given."""
    config_path = Path(path) if path else DEFAULT_CONFIG
    if not config_path.is_file():
        if path:
            raise FileNotFoundError(f"Agent config not found: {config_path}")
        return AgentConfig()
    data = _parse_config_text(config_path.read_text())
    known = {field: data[field] for field in AgentConfig.__dataclass_fields__
             if field in data}
    return AgentConfig(**known)


def resolve_agent_config(config_path=None, backend=None, model=None,
                         effort=None, max_tokens=None) -> AgentConfig:
    """File defaults, then explicit overrides (CLI / function kwargs)."""
    cfg = load_agent_config(config_path)
    if backend is not None:
        cfg.backend = backend
    if model is not None:
        cfg.model = model
    if effort is not None:
        cfg.effort = effort
    if max_tokens is not None:
        cfg.max_tokens = max_tokens
    return AgentConfig(**cfg.__dict__)


# ============================================================
# Data loading (cached)
# ============================================================

@lru_cache(maxsize=1)
def _genome() -> str:
    """Load the H37Rv reference sequence as one uppercase string."""
    if not GENOME_FASTA.exists():
        return ""
    parts = []
    with open(GENOME_FASTA) as fh:
        for line in fh:
            if not line.startswith(">"):
                parts.append(line.strip())
    return "".join(parts).upper()


@lru_cache(maxsize=1)
def _matrix() -> pd.DataFrame:
    """Load the ML feature matrix (9,798 isolates x 2,693 positions + labels)."""
    if not ML_MATRIX.exists():
        return pd.DataFrame()
    return pd.read_csv(ML_MATRIX)


def _reverse_complement(seq: str) -> str:
    return "".join(COMPLEMENT.get(b, "N") for b in reversed(seq))


def _gene_orf(gene: str) -> str:
    """Return the coding sequence for a gene, 5'->3' on its own strand."""
    start, end = AMR_GENES[gene]
    sub = _genome()[start - 1:end]
    return sub if GENE_STRAND[gene] == "+" else _reverse_complement(sub)


def _translate(codon: str) -> str:
    return CODON_TABLE.get(codon, "?")


@lru_cache(maxsize=1)
def _validate_reading_frames() -> bool:
    """
    Verify the strand/frame table before any residue is reported.

    Two independent checks:
      1. Every gene ORF is divisible by 3, ends in a stop codon, and has no
         internal stop -- this confirms strand and frame.
      2. The three FRAME_ANCHORS reproduce their known codon and residue.

    Returns False if either fails, in which case lookup_position degrades to
    reporting gene and coordinate only.
    """
    if not _genome():
        return False

    for gene in AMR_GENES:
        if gene not in GENE_STRAND:
            return False
        orf = _gene_orf(gene)
        if len(orf) == 0 or len(orf) % 3 != 0:
            return False
        protein = "".join(_translate(orf[i:i + 3]) for i in range(0, len(orf), 3))
        if not protein.endswith("*") or "*" in protein[:-1]:
            return False

    for pos, gene, exp_codon, exp_aa in FRAME_ANCHORS:
        codon_num, _ = _codon_index(pos, gene)
        if codon_num != exp_codon:
            return False
        orf = _gene_orf(gene)
        codon = orf[(codon_num - 1) * 3:(codon_num - 1) * 3 + 3]
        if _translate(codon) != exp_aa:
            return False

    return True


def _codon_index(pos: int, gene: str):
    """Return (1-based codon number, 0-based offset within the codon)."""
    start, end = AMR_GENES[gene]
    offset = (pos - start) if GENE_STRAND[gene] == "+" else (end - pos)
    return offset // 3 + 1, offset % 3


def _parse_position(position) -> int:
    """Accept 'pos_761155', '761155', or 761155."""
    return int(str(position).replace("pos_", "").strip())


def _gene_for(pos: int):
    for gene, (start, end) in AMR_GENES.items():
        if start <= pos <= end:
            return gene
    return None


# ============================================================
# Tools
# ============================================================

def lookup_position(position: str, observed_nucleotide: str = "") -> dict:
    """Identify the gene, codon, and amino-acid change at a genomic position.

    Args:
        position: Genomic coordinate, e.g. "pos_761155" or "761155".
        observed_nucleotide: Optional observed base (A/T/C/G) so the resulting
            amino-acid substitution can be resolved. Omit if unknown.
    """
    pos = _parse_position(position)
    gene = _gene_for(pos)
    if gene is None:
        return {"position": pos, "gene": None,
                "note": "Position is outside the nine AMR genes in the feature set."}

    start, end = AMR_GENES[gene]
    result = {
        "position": pos, "gene": gene, "strand": GENE_STRAND[gene],
        "gene_span": f"{start}-{end}",
    }

    if not _validate_reading_frames():
        result["residue"] = None
        result["note"] = (
            "Reading-frame validation failed, so no amino-acid call is made. "
            "Gene and coordinate are reliable; the residue is unresolved."
        )
        return result

    codon_num, offset = _codon_index(pos, gene)
    orf = _gene_orf(gene)
    ref_codon = orf[(codon_num - 1) * 3:(codon_num - 1) * 3 + 3]
    ref_aa = _translate(ref_codon)

    result.update({
        "codon_number": codon_num, "reference_codon": ref_codon,
        "reference_aa": ref_aa,
    })

    obs = str(observed_nucleotide or "").strip().upper()
    if obs in ("A", "T", "C", "G"):
        # The feature encodes the base on the forward strand; complement it
        # for a gene transcribed from the reverse strand.
        coding_base = obs if GENE_STRAND[gene] == "+" else COMPLEMENT[obs]
        alt_codon = ref_codon[:offset] + coding_base + ref_codon[offset + 1:]
        alt_aa = _translate(alt_codon)
        result.update({
            "observed_nucleotide": obs, "alternate_codon": alt_codon,
            "alternate_aa": alt_aa, "synonymous": alt_aa == ref_aa,
            "substitution": f"{gene} {ref_aa}{codon_num}{alt_aa}",
        })
    else:
        result["substitution"] = f"{gene} {ref_aa}{codon_num}?"
        result["note"] = ("No observed nucleotide supplied, so only the reference "
                          "residue is resolved.")
    return result


def cohort_frequency(position: str, drug: str) -> dict:
    """How often a position is mutated in resistant vs susceptible isolates.

    Computed over the 9,798-isolate training cohort. Use this to tell a common
    resistance driver from an isolate-specific variant.

    Args:
        position: Genomic coordinate, e.g. "pos_761155".
        drug: One of RIFAMPICIN, ISONIAZID, ETHAMBUTOL, PYRAZINAMIDE.
    """
    drug = drug.strip().upper()
    if drug not in DRUGS:
        return {"error": f"Unknown drug {drug!r}. Expected one of {DRUGS}."}

    df = _matrix()
    if df.empty:
        return {"error": "Feature matrix not available locally."}

    col = f"pos_{_parse_position(position)}"
    if col not in df.columns:
        return {"position": col, "drug": drug,
                "note": "Position is not one of the 2,693 features in the matrix."}

    sub = df[[col, drug]].dropna(subset=[drug])
    resistant = sub[sub[drug] == 1]
    susceptible = sub[sub[drug] == 0]
    if len(resistant) == 0 or len(susceptible) == 0:
        return {"error": f"No labelled isolates for {drug}."}

    r_mut = int((resistant[col] != 0).sum())
    s_mut = int((susceptible[col] != 0).sum())
    return {
        "position": col, "drug": drug,
        "resistant_isolates": len(resistant),
        "resistant_mutated": r_mut,
        "resistant_mutated_pct": round(100 * r_mut / len(resistant), 1),
        "susceptible_isolates": len(susceptible),
        "susceptible_mutated": s_mut,
        "susceptible_mutated_pct": round(100 * s_mut / len(susceptible), 1),
    }


def cross_drug_context(position: str) -> dict:
    """Whether a position's SHAP attribution shifts under co-resistance.

    Returns rows from the six pairwise drug analyses. A large shift means the
    model weighted this position differently in co-resistant isolates than in
    isolates resistant to one drug only.

    Args:
        position: Genomic coordinate, e.g. "pos_2155168".
    """
    col = f"pos_{_parse_position(position)}"
    rows = []
    for path in sorted(CROSS_DRUG_DIR.glob("shap_cross_drug_*.csv")):
        pair = path.stem.replace("shap_cross_drug_", "")
        try:
            df = pd.read_csv(path)
        except Exception:
            continue
        hit = df[df["position"] == col]
        if hit.empty:
            continue
        r = hit.iloc[0]
        a, b = pair.split("_")
        rows.append({
            "pair": f"{a} vs {b}", "gene": r.get("gene"),
            f"{a}_only": round(float(r[f"shap_{a}_only"]), 5),
            f"{b}_only": round(float(r[f"shap_{b}_only"]), 5),
            f"{a}_coresistant": round(float(r[f"shap_{a}_in_coresist"]), 5),
            f"{b}_coresistant": round(float(r[f"shap_{b}_in_coresist"]), 5),
            "max_abs_shift": round(float(r["max_abs_delta"]), 5),
        })

    if not rows:
        return {"position": col,
                "note": "Position does not appear in any cross-drug analysis."}
    return {"position": col, "pairs": rows,
            "reminder": ("A shift reflects how the model weighted this feature in "
                         "co-resistant isolates. Co-occurrence in MDR strains is the "
                         "usual explanation, not a shared mechanism.")}


def get_shap_detail(drug: str, top_n: int = 40) -> dict:
    """Full SHAP attribution table for one drug, beyond the top 20 in the summary.

    Args:
        drug: One of RIFAMPICIN, ISONIAZID, ETHAMBUTOL, PYRAZINAMIDE.
        top_n: How many features to return, ranked by absolute SHAP value.
    """
    drug = drug.strip().upper()
    active = _ACTIVE.get()
    if active is None:
        return {"error": "No prediction file is loaded."}
    if drug not in DRUGS:
        return {"error": f"Unknown drug {drug!r}. Expected one of {DRUGS}."}

    csv_path = active["dir"] / f"{active['sample_id']}_{drug}_shap_values.csv"
    if not csv_path.exists():
        entry = active["data"].get(drug, {})
        feats = entry.get("top_shap_features", [])[:top_n]
        return {"drug": drug, "source": "summary JSON (top 20 only)",
                "features": feats}

    df = pd.read_csv(csv_path)
    df = df.reindex(df["shap_value"].abs().sort_values(ascending=False).index)
    out = df.head(int(top_n)).to_dict(orient="records")
    for row in out:
        row["nucleotide"] = NUC_DECODE.get(int(row.get("encoded_value", 0)), "?")
        row["shap_value"] = round(float(row["shap_value"]), 6)
    return {"drug": drug, "source": str(csv_path.name),
            "total_features": len(df), "returned": len(out), "features": out}


TOOL_FUNCTIONS = [lookup_position, cohort_frequency, cross_drug_context, get_shap_detail]
TOOL_BY_NAME = {fn.__name__: fn for fn in TOOL_FUNCTIONS}


# ============================================================
# Prediction file handling
# ============================================================

def load_predictions(path) -> dict:
    """Load a *_predictions.json produced by vcf_to_prediction.py."""
    path = Path(path)
    with open(path) as fh:
        data = json.load(fh)
    sample_id = next(
        (v.get("sample") for v in data.values() if isinstance(v, dict) and v.get("sample")),
        path.stem.replace("_predictions", ""),
    )
    return {"data": data, "sample_id": sample_id, "dir": path.parent, "path": path}


def _who_note(feat: dict) -> str:
    """WHO catalogue context for one attributed position, or '' if there is none.

    Three cases, in order of strength: the allele itself is graded; the allele
    is in the catalogue but ungraded for this drug while graded for another;
    the isolate carries the reference base at a site where graded alleles
    exist. The third is what explains a susceptible call.
    """
    cat = feat.get("who_catalogue") or {}

    entries = cat.get("entries") or []
    if entries:
        e = entries[0]
        return f"  [WHO: {e.get('variant')}, {e.get('final_confidence_grading')}]"

    others = feat.get("who_other_drug_associations") or []
    if others:
        o = others[0]
        return (f"  [WHO: {o.get('variant')}, no grading for this drug; "
                f"{o.get('final_confidence_grading')} for {o.get('drug')}]")

    hits = cat.get("coordinate_hits") or []
    if hits:
        return f"  [WHO: {hits[0].get('variant')}, ungraded]"

    sites = feat.get("catalogue_site_alleles") or []
    if sites:
        return (f"  [WHO site: {', '.join(sites[:3])} — "
                f"isolate carries the reference base]")

    return ""


def format_profile(active: dict, drug: str = None) -> str:
    """Render the resistance profile and top attributions as prompt text.

    Carries the WHO catalogue grounding written by the prediction step, so the
    model is given it rather than having to fetch it with get_shap_detail.
    """
    lines = []
    drugs = [drug.upper()] if drug else [d for d in DRUGS if d in active["data"]]
    for d in drugs:
        entry = active["data"].get(d)
        if not entry:
            continue
        lines.append(
            f"\n{d}: {entry.get('prediction')} "
            f"(P(resistant) = {entry.get('probability_resistant')})"
        )
        for feat in entry.get("top_shap_features", [])[:8]:
            enc = feat.get("encoded_value", 0)
            lines.append(
                f"  {feat.get('position')}  {feat.get('gene')}  "
                f"base={NUC_DECODE.get(int(enc), '?')}  "
                f"SHAP={feat.get('shap_value'):+.5f}"
                f"{_who_note(feat)}"
            )

        known = entry.get("known_resistance_variants") or []
        if known:
            lines.append(f"  WHO alleles graded for {d} in this isolate:")
            for k in known:
                ppv = k.get("ppv")
                ppv_txt = f", PPV {ppv:.3f}" if isinstance(ppv, (int, float)) else ""
                lines.append(
                    f"    {k.get('variant')} ({k.get('effect')})"
                    f" — {k.get('final_confidence_grading')}{ppv_txt}"
                )
        else:
            lines.append(f"  No WHO allele graded 1 or 2 for {d} in this isolate.")
    return "\n".join(lines) if lines else "(no drug entries found)"


# ============================================================
# Agent
# ============================================================

def _user_prompt(active, drug, question) -> str:
    profile = format_profile(active, drug)
    if question:
        return QUESTION_REQUEST.format(
            sample_id=active["sample_id"], profile=profile, question=question
        )
    return REPORT_REQUEST.format(sample_id=active["sample_id"], profile=profile)


def interpret(predictions_path, **kwargs):
    """Run the interpretation agent over a prediction JSON file.

    Args:
        predictions_path: Path to a *_predictions.json file.
        drug: Restrict the profile to one drug. None covers all four.
        question: Free-text question. None produces a full report.
        trace: Print each tool call as it happens.
        config_path: YAML/JSON file with backend and model. Defaults to
            scripts/agent/agent_config.yaml.
        backend: anthropic or openai. Overrides the config file.
        model: Provider model id. Overrides the config file.
        effort: Anthropic reasoning effort (low | medium | high | xhigh | max).

    Returns:
        The agent's written interpretation as a string.
    """
    return _run_agent(load_predictions(predictions_path), **kwargs)


def interpret_results(results, sample_id, output_dir=None, **kwargs):
    """Run the agent over an in-memory results dict.

    Same shape as the prediction JSON — {DRUG: {prediction, probability_resistant,
    top_shap_features}}. Lets the dashboard call the agent without writing a
    temp file. Accepts the same keyword arguments as interpret().
    """
    active = {"data": results, "sample_id": sample_id,
              "dir": Path(output_dir) if output_dir else Path("."), "path": None}
    return _run_agent(active, **kwargs)


def _run_agent(active, drug=None, question=None, trace=False,
               model=None, effort=None, backend=None, config=None,
               config_path=None, max_tokens=None):
    """Shared agent loop for interpret() and interpret_results()."""
    cfg = config if isinstance(config, AgentConfig) else resolve_agent_config(
        config_path=config_path, backend=backend, model=model,
        effort=effort, max_tokens=max_tokens,
    )
    token = _ACTIVE.set(active)
    try:
        prompt = _user_prompt(active, drug, question)
        print(f"  Agent: {cfg.backend} / {cfg.model}", file=sys.stderr)
        if cfg.backend == "openai":
            return _run_openai(cfg, prompt, trace)
        return _run_anthropic(cfg, prompt, trace)
    finally:
        _ACTIVE.reset(token)


def _run_anthropic(cfg: AgentConfig, prompt: str, trace: bool) -> str:
    try:
        import anthropic
        from anthropic import beta_tool
    except ImportError as exc:
        raise ImportError(
            "The anthropic package is required for backend=anthropic:\n"
            "    pip install anthropic"
        ) from exc

    client = anthropic.Anthropic()
    runner = client.beta.messages.tool_runner(
        model=cfg.model,
        max_tokens=cfg.max_tokens,
        thinking={"type": "adaptive"},
        output_config={"effort": cfg.effort},
        system=SYSTEM_PROMPT,
        tools=[beta_tool(fn) for fn in TOOL_FUNCTIONS],
        messages=[{"role": "user", "content": prompt}],
    )

    final_text = []
    for message in runner:
        for block in message.content:
            if block.type == "tool_use" and trace:
                print(f"  [tool] {block.name}({json.dumps(block.input)})",
                      file=sys.stderr)
            elif block.type == "text":
                final_text = [block.text]
    return "\n".join(final_text).strip()


def _openai_tools():
    """JSON-schema tool list derived from the Python function signatures."""
    type_map = {int: "integer", float: "number", bool: "boolean"}
    tools = []
    for fn in TOOL_FUNCTIONS:
        properties = {}
        required = []
        for name, param in inspect.signature(fn).parameters.items():
            if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
                continue
            json_type = type_map.get(param.annotation, "string")
            properties[name] = {"type": json_type}
            if param.default is inspect.Parameter.empty:
                required.append(name)
        tools.append({
            "type": "function",
            "function": {
                "name": fn.__name__,
                "description": (fn.__doc__ or "").strip(),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        })
    return tools


def _run_openai(cfg: AgentConfig, prompt: str, trace: bool) -> str:
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise ImportError(
            "The openai package is required for backend=openai:\n"
            "    pip install openai"
        ) from exc

    client = OpenAI()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": prompt},
    ]
    tools = _openai_tools()

    for _ in range(25):
        response = client.chat.completions.create(
            model=cfg.model,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=cfg.max_tokens,
        )
        message = response.choices[0].message
        assistant = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            assistant["tool_calls"] = [
                {
                    "id": call.id,
                    "type": "function",
                    "function": {
                        "name": call.function.name,
                        "arguments": call.function.arguments,
                    },
                }
                for call in message.tool_calls
            ]
        messages.append(assistant)

        if not message.tool_calls:
            return (message.content or "").strip()

        for call in message.tool_calls:
            fn = TOOL_BY_NAME.get(call.function.name)
            try:
                args = json.loads(call.function.arguments or "{}")
            except json.JSONDecodeError:
                args = {}
            if trace:
                print(f"  [tool] {call.function.name}({json.dumps(args)})",
                      file=sys.stderr)
            if fn is None:
                result = {"error": f"Unknown tool {call.function.name!r}."}
            else:
                result = fn(**args)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps(result, default=str),
            })

    raise RuntimeError("OpenAI agent exceeded the tool-call limit (25 rounds).")


def credentials_available(backend=None, config_path=None) -> bool:
    """True if the selected backend's SDK and credentials are resolvable."""
    cfg = resolve_agent_config(config_path=config_path, backend=backend)
    if cfg.backend == "openai":
        try:
            import openai  # noqa: F401
        except ImportError:
            return False
        return bool(os.environ.get("OPENAI_API_KEY"))

    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True
    cfg_dir = Path(os.environ.get("ANTHROPIC_CONFIG_DIR",
                                  Path.home() / ".config" / "anthropic"))
    return (cfg_dir / "credentials").exists()


def credential_help(cfg: AgentConfig) -> str:
    if cfg.backend == "openai":
        return (
            "No OpenAI credentials found.\n"
            "  export OPENAI_API_KEY=...\n"
            "The tools in this module work offline; only the agent needs a key."
        )
    return (
        "No Anthropic credentials found.\n"
        "  export ANTHROPIC_API_KEY=...   (or run: ant auth login)\n"
        "The tools in this module work offline; only the agent needs a key."
    )


# ============================================================
# CLI
# ============================================================

def main():
    ap = argparse.ArgumentParser(
        description="Interpret FORUM-TB SHAP attributions with a tool-using agent.")
    ap.add_argument("--predictions", required=True,
                    help="Path to a *_predictions.json file.")
    ap.add_argument("--drug", choices=DRUGS, help="Restrict to one drug.")
    ap.add_argument("--question", help="Ask a specific question instead of a full report.")
    ap.add_argument("--trace", action="store_true", help="Print tool calls to stderr.")
    ap.add_argument("--config", default=str(DEFAULT_CONFIG),
                    help="YAML/JSON file with backend and model "
                         f"(default: {DEFAULT_CONFIG}).")
    ap.add_argument("--backend", choices=["anthropic", "claude", "openai"],
                    help="Override the config file backend.")
    ap.add_argument("--model", help="Override the config file model id.")
    ap.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"],
                    help="Anthropic reasoning effort. Overrides the config file.")
    ap.add_argument("--output", help="Write the report to a file as well as stdout.")
    args = ap.parse_args()

    cfg = resolve_agent_config(
        config_path=args.config, backend=args.backend,
        model=args.model, effort=args.effort,
    )
    if not credentials_available(backend=cfg.backend, config_path=args.config):
        sys.exit(credential_help(cfg))

    report = interpret(
        args.predictions, drug=args.drug, question=args.question,
        trace=args.trace, config=cfg,
    )
    print(report)
    if args.output:
        Path(args.output).write_text(report)
        print(f"\n[saved to {args.output}]", file=sys.stderr)


if __name__ == "__main__":
    main()
