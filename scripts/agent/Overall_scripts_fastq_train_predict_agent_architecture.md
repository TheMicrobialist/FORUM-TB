## Overall plan

Same as `SHAP-mTB-AMR/scripts/agent/shap_agent_architecture.md`.

Phases 1–2 run once, offline, and their outputs are frozen. Phases 3–4 run per isolate and reuse those frozen outputs unchanged.

- Phase 1. build the dataset (once, offline)
  - turns raw sequencing reads from public databases into one table the models can train on
  - output is `resistance_dataset/ml_matrix.csv.gz`, and everything downstream depends on its exact shape

  - Step 1. Download reads
    - file: `scripts/download_fastq.sh`
    - tool: sra-tools, from ENA / SRA
  - Step 2. Quality control
    - file: `scripts/qc_fastq.sh`
    - tool: fastp 1.0.1
  - Step 3. Align reads to the reference genome
    - file: `scripts/align_tb.sh`
    - tool: BWA-MEM 0.7.18
    - ref: H37Rv, `NC_000962.3` (4.4 Mb)
  - Step 4. Call and filter variants — this is what produces the filtered VCF
    - files: `scripts/call_variants.sh`, `scripts/qc_variants.py`
    - tool: bcftools 1.22, filter QUAL >= 20 and MQ >= 30
  - Step 5. Build the feature matrix
    - file: `scripts/vcf_to_matrix.py`
    - start: 705,226 unique variant positions genome-wide across all isolates
    - keep SNPs only — indels are dropped here and never return in 3a
    - keep positions inside the 9 AMR genes only
    - drop 3 isolates with >10,000 variants each (ERR1034652, ERR1035353, ERR1213920), likely contamination or platform artefacts
    - encode each kept position as one integer column: 0=REF 1=A 2=T 3=C 4=G
    - out: `resistance_dataset/ml_matrix.csv.gz` — 9,798 isolates x 2,693 positions, plus 4 drug label columns. In plain terms, 705,226 positions in and 2,693 out: the 9 genes add up to 19,170 bp across all nine, and only 2,693 of those positions (14%) ever varied, so the rest carry no information

- Phase 2. train (once, offline)
  - fits one Random Forest per drug on the matrix from Phase 1, then saves them. Phase 3 reuses these files as-is

  - Step 1. Split
    - 80/20 train/test, `random_state=42`, stratified by label (`rf_shap_v2.py:66-70`)
  - Step 2. Fit and save
    - file: `scripts/rf_shap_v2.py`
    - model: Random Forest, 300 trees, balanced class weights
    - fits on the 80% train split only, then dumps that model (`:81`, `:85`)
    - out: `models/rf_RIFAMPICIN_v2.joblib`, and the same for INH / EMB / PZA
    - note: 244 MB total, not in git, hosted on HuggingFace
  - also: `scripts/benchmark.py` compared 7 classifiers; RF won every drug
  - watch: because the saved model saw the 80%, any isolate from that split scores in-sample. ERR040120 (the example genome) is in the train split for all four drugs

- Phase 3. predict one isolate
  - all of it runs from `scripts/vcf_to_prediction.py`
  - Shared input: one filtered VCF, already aligned and called by Phase 1
    - example: `test_data/ERR040120.filtered.vcf.gz`
    - one parse, two different sets (`:145`):
      - 944 SNPs — go to 3a, the model only accepts `len(REF)==1 and len(ALT)==1`, meaning both the reference and the alternate allele are a single nucleotide. A one-letter swap like C to T. Anything longer on either side is an insertion or deletion and is rejected here
      - 1,019 alleles — go to 3b. The catalogue takes everything: the same 944 SNPs plus 75 insertions and deletions. Those 75 have no column in 3a and never reach the model, so 3b is the only path that sees them, and WHO can still return a graded verdict on them

  - 3a. Random Forest + SHAP
    - for the target genome, predicts resistant or susceptible for each of the 4 drugs (rifampicin, isoniazid, ethambutol, pyrazinamide), then ranks which SNP positions drove each prediction. The Random Forest makes the call; SHAP then explains which SNPs pushed it hardest
    - both steps reuse Phase 2 unchanged — the 4 trained forests, the 2,693-column input schema, the 100-isolate SHAP background. Nothing is retrained per sample
    - the limit of that: a resistance SNP at a position that never varied in training has no column, so the model cannot see it however important it is. Only retraining adds columns. 3b is what catches these

    - Two numbers, taking ERR040120 as the example
      - 944 SNPs — read from ERR040120's own VCF. Nucleotide positions where this sample differs from the reference H37Rv (4.4 Mb)
      - 2,693 columns — read from the training matrix `resistance_dataset/ml_matrix.csv.gz`. The 9 AMR genes add up to 19,170 bp across all nine, so 19,170 positions in total. Across the 9,798 training isolates, 2,693 of them (14%) carried a SNP in at least one isolate. Those 2,693 became the columns; the other 16,477 never varied, carry no information, and were dropped
      - all four drug models read these same 2,693 columns. The 4 drugs are what the model outputs, not part of what it reads

    - Step 1. Encode: fit the 944 onto the 2,693
      - only SNPs landing on one of the 2,693 count. For ERR040120, 5 of 944 (`vcf_to_prediction.py:145`). In plain terms, the model has a fixed list of 2,693 positions it knows about: a SNP in this sample is recorded only if it sits on one of them, and anywhere else there is no slot to put it in
      - the other 939 fall outside the 9 genes, have no column, and are dropped
      - the remaining 2,688 columns are 0, meaning "same as H37Rv"
      - a column is a position, not a mutation: 0=REF 1=A 2=T 3=C 4=G, so C>T and C>G at one position share a column
      - result: a 2,693-long vector, 5 non-zero. In plain terms, the model always gets the same 2,693 slots whatever the sample; for ERR040120, 5 slots say "a mutation is here" and the other 2,688 say "matches the reference"

    - Step 2. Score
      - one Random Forest per drug, all reading that same vector
      - `predict_proba >= 0.5` means "Resistant" (`:206-207`)
      - caveat: ERR040120 is in the training set for all four drugs, so RIF 1.0000 and INH 1.0000 are in-sample, not held-out. `table6_case_study.tex` and `results.tex:67` are built on it

    - Step 3. Attribute
      - SHAP `TreeExplainer` for tree models
      - interventional mode, drop exact zeros, keep top 20 (`:211-234`)
      - 100-isolate background: 100 isolates sampled from the training matrix (`:213-217`). They are the baseline — a SHAP value says how far that position pushed the score away from the average prediction across those 100
      - returns position plus signed value, e.g. `pos_761155, +0.280542`. Naming it rpoB S450L happens in 3b or Phase 4
      - `nonzero_shap()` drops exactly-zero features, so a missing name reads back as 0.0, meaning that position contributed nothing to the score

  - 3b. WHO catalogue, by Noah
    - takes every allele from the same VCF and asks a curated reference whether it is a known resistance mutation. Independent of our model — it does not read the forests, the 2,693 columns, or any SHAP value
    - files: `scripts/who_catalogue.py`, `scripts/WHO-UCN-TB-2023.7-eng.xlsx`
    - added: Noah, 2026-10-02, issue #20
    - source: `scripts/Instruction of use for incorporation of the mutations catalogue version 2 results into bioinformatic pipeline.pdf`

    - Step 1. Find the graded-variant name
      - sheet `Genomic_coordinates`, header row 1
      - exact match on (position, REF, ALT). CHROM is dropped because it is always `NC_000962.3`
      - the index is 118,431 alleles keyed `(int(pos), REF.upper(), ALT.upper())`, with the same normalisation applied on both build and lookup. Faithful to the WHO instruction PDF
      - verified live 2026-10-18: `761155 C>T` for rifampicin returns `rpoB_p.Ser450Leu`, grade `1) Assoc w R`
    - Step 2. Look up its grade
      - sheet `Catalogue_master_file`, header row 3
      - exact match on the graded-variant name from Step 1, giving the per-drug confidence grade. In plain terms, Step 1 turns coordinates into WHO's name for the mutation, and Step 2 turns that name into WHO's verdict on it

    - many-to-many both ways: one allele can map to several graded variants and vice versa (`who_catalogue.py:17-18`; `:154` returns a list)
    - gives: mutation name, WHO confidence grade, plus group 1/2 grades for other drugs when the queried drug has no row (`:148`)
    - what it can see: all 1,019 alleles, so promoter SNPs and indels outside the 2,693 columns still get flagged
    - slow: about 51 seconds per process. Check against #11, and whether it can be accelerated
    - caveat: WHO provides no coordinates for deletion graded-variants (e.g. `pncA_deletion` — and pncA is one of our nine genes, PZA our weakest drug), omits LoF variants never graded group 1 or 2, and links only observed varying-length variants
    - caveat: WHO normalizes its coordinates with `bcftools norm` and says indel matching is guaranteed only if ours are normalized too. `who_catalogue.py` does not normalize. Harmless for SNPs; may silently miss some of those 75 non-SNP alleles. Check whether `bcftools norm` ran upstream before claiming indel coverage in the paper

  - Output: two files, and both now hold 3a and 3b

    - `results/predictions/<SAMPLE>_predictions.json` — one record per drug, where 3a and 3b merge (`:271-278`)
      - from 3a: `prediction`, `probability_resistant`, `top_shap_features`
      - from 3b: `who_catalogue_source`, `known_resistance_variants`, `explanation`
    - `results/predictions/<SAMPLE>_<DRUG>_shap_values.csv` — the per-position table, one row per feature. Columns are `position, gene, encoded_value, shap_value` from 3a, plus `who_variant, who_mutation, who_effect, who_grading` from 3b

    - status: both examples are complete as of 2026-10-18, all four drugs with 3a and 3b
      - ERR040120 — MDR. Resistant to all four. WHO group 1 variants: `rpoB_p.Ser450Leu`, `katG_p.Ser315Thr`, `embB_p.Gly406Asp`
      - ERR038266 — pan-susceptible. Susceptible to all four, no WHO group 1 or 2 alleles, which is the expected result. Its explanations turn on resistance sites carrying the reference base rather than on mutations being present
      - both match the ground-truth labels 4/4. Both are in the training split, so in-sample

- Phase 4. interpret (the agent)
  - A bespoke tool-using harness around an LLM. "Harness" is informal, not a standard with types — named frameworks exist (LangChain, OpenAI Agents SDK, smolagents), but this one is hand-written Python that delegates only the loop to Anthropic's SDK `beta.messages.tool_runner`
    - Harness: `scripts/agent/shap_agent.py` — a system prompt, four tool definitions, and the loop that runs them. It decides what the model may look up and forces it to look rather than recall, which is the interpretability claim
    - Model: set in `scripts/agent/agent_config.yaml` as backend / model / effort / max_tokens, now anthropic / `claude-opus-5` / high / 16000. `openai` also works (`_run_openai:684`)

  - Step 1. Load the Phase 3 record, 3a and 3b together
    - `load_predictions` (`:474`) reads the whole of `<SAMPLE>_predictions.json` into a dict — every drug, every key, both halves
    - example: `results/predictions/ERR040120_predictions.json`

  - Step 2. Build the profile that goes into the prompt
    - `format_profile` renders one block per drug, preceded by a Provenance line when `_in_training_cohort` matches the sample against `ml_matrix.csv.gz`
    - provenance reports cohort membership, not train/test split: reproducing the split would couple the agent to `rf_shap_v2.py`'s constants and misreport silently if they changed
    - from 3a: prediction, P(resistant), and the top 8 of `top_shap_features` as position / gene / base / SHAP value
    - from 3b, via `_who_note` (`:486`): each position's WHO grade, or the graded alleles at that site when the isolate carries the reference base; then the drug's `known_resistance_variants` with PPV, or a line saying there are none
    - `_user_prompt` (`:512`) picks the template on the `--question` flag: absent gives `REPORT_REQUEST`, the four-drug report and the default; present gives `QUESTION_REQUEST`. `--drug` narrows the profile to one drug either way
    - `SYSTEM_PROMPT` supplies the rules: the five WHO grades, the interpretive guardrails, and the required output shape
    - call settings: `thinking={"type": "adaptive"}`, `output_config.effort` from the config (now high), `max_tokens` 16000

  - Step 3. Run the tool loop
    - `_run_agent` (`:553`) resolves the config and dispatches to `_run_anthropic` (`:572`) or `_run_openai` (`:684`)
    - anthropic path: the SDK's `beta.messages.tool_runner` drives call → tool → call; the code iterates messages and keeps the text
    - openai path: a hand-rolled loop over `chat.completions`, capped at 25 iterations
    - four tools, which the model must call rather than answer from memory. Only `get_shap_detail` reads this isolate's Phase 3 output; the rest supply cohort and reference context
      - `lookup_position` (`:295`) — gene, codon, amino-acid change, computed from `reference/H37Rv.fasta` (Phase 1)
      - `cohort_frequency` (`:352`) — mutated in resistant vs susceptible across the 9,798 isolates, from `ml_matrix.csv.gz` (Phase 1)
      - `cross_drug_context` (`:394`) — whether attribution shifts under co-resistance, from `results/shap_cross_drug_*.csv` (the separate analysis in #8)
      - `get_shap_detail` (`:435`) — the per-drug CSV beyond the top 20. Every column, so 3a's SHAP and 3b's WHO fields both
    - `_validate_reading_frames` (`:235`, cached) runs on the first `lookup_position`: all nine ORFs must divide by 3 and rpoB S450L / katG S315T / embB M306 must reproduce. On failure the residue is dropped and gene plus coordinate still returned (`:315-320`)

  - Step 4. Output
    - returns the final text block only (`:600`) — the answer, not the running commentary
    - `--trace` prints each tool call to stderr as it happens
    - required shape per drug: `## <DRUG>` heading, then Evidence (3a's SHAP with 3b's grade per position), Catalogue agreement (3b), Confidence, Caveats
    - status: first live run 2026-10-18 on openai / gpt-4o, #36. Never run on the anthropic backend

  - Second run
    - all four drugs carry the provenance caveat; no "Caveats: None"
    - provenance re-checked: 12/12 evidence lines, no mismatch
    - determinism at temperature 0: two consecutive runs differ in 4 lines of 64, both Confidence sentences that are semantically identical. Every evidence line, SHAP value, grade, Catalogue agreement and Caveat is byte-identical. The honest claim is near-deterministic, not deterministic
    - tool calls collapsed to four `lookup_position` and nothing else — down from 13 and 8. The richer profile means the model investigates less
    - consequence to decide for #38: `cross_drug_context` went unused on the isolate where the cross-drug result is the interesting finding. An earlier run called it three times and reached the population-structure conclusion; this run states PZA from the profile alone. The framing "the agent is forced to look things up" needs rewording — the profile carries verified facts, the tools cover what it does not

    - commands, from the repo root with the env on PATH (see Environment)

          export OPENAI_API_KEY=...

          python scripts/agent/shap_agent.py \
              --predictions results/predictions/ERR040120_predictions.json \
              --backend openai --model gpt-4o --trace

      determinism check, two runs diffed

          python scripts/agent/shap_agent.py --predictions results/predictions/ERR040120_predictions.json \
              --backend openai --model gpt-4o > /tmp/run_a.txt 2>&1
          python scripts/agent/shap_agent.py --predictions results/predictions/ERR040120_predictions.json \
              --backend openai --model gpt-4o > /tmp/run_b.txt 2>&1
          diff /tmp/run_a.txt /tmp/run_b.txt

  - next: structured output (`output_config.format`)
    - constrain the response to a schema — per drug a verdict, confidence, an evidence array (position, gene, SHAP, WHO grade, which tool supplied it), caveats. The API enforces validity instead of the prompt asking for it
    - the point is that faithfulness becomes testable: assert programmatically that every grade the model reports matches what a tool returned. "Verified across N isolates" is a result; "we read it and it looked right" is not
    - unverified: whether structured outputs compose with `beta.messages.tool_runner`. If they do not, this means replacing the SDK loop with a manual one — a day, not an hour. Check before committing to it
    - subsumes the skeleton above rather than replacing it: the sections become schema fields, so nothing done now is wasted

  - still open: overlap with Phase 3, which already writes a deterministic `explanation` per drug with the same WHO grounding. Less pressing now that the agent is handed the same facts, but the paper should say which one it is showing

## Environment

conda env `forum-tb`, Python 3.11.13. Activate it, or call its python by full path. The models load with no version warnings because scikit-learn matches the version that pickled them exactly.

    numpy==2.3.5          scikit-learn==1.6.1    scipy==1.17.1
    pandas==3.0.6         shap==0.51.0           joblib==1.6.0
    openpyxl==3.1.5       PyYAML==6.0.3

`scikit-learn==1.6.1` is the pin that matters: the `.joblib` files carry `_sklearn_version = 1.6.1` and reference `numpy._core`, the numpy >= 2.0 layout. Loading across sklearn versions is unsupported and can silently reconstruct a different estimator, so it is pinned exactly rather than loosely.

The anaconda base env (Python 3.9.7, numpy 1.20.3, scikit-learn 0.24.2) cannot load the models and was deliberately left untouched.

Run anything through:

```bash
python scripts/vcf_to_prediction.py \
    --vcf test_data/ERR040120.filtered.vcf.gz --all-drugs
```

Offline tool checks need none of this:

```bash
python3 scripts/agent/test_shap_agent_tools.py      # expect 37 passed, 0 failed
```

This pin list is what #35 asks to be put in the repo. It is not there yet — it exists only here.
