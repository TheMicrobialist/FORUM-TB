# Agent output

Phase 4 output for ERR040120, the MDR demo isolate. Produced on 2026-10-18 with
backend `openai` / `gpt-4o`, `temperature=0`.

| File | What |
|---|---|
| `ERR040120_report.txt` | the written interpretation, four drugs |
| `ERR040120_trace.txt` | every tool call the agent made, from `--trace` |
| `ERR040120_determinism_run_a.txt` | two consecutive runs at temperature 0, |
| `ERR040120_determinism_run_b.txt` | diffed to show what varies |

The trace is the point: every SHAP value and WHO grade in the report should
trace either to a `[tool]` line here or to the profile the agent was given.
Checked for this run — 13 evidence lines, no mismatches.

The determinism pair differs in 4 lines of 64, both Confidence sentences that
say the same thing in different words. Near-deterministic, not deterministic.

Reproduce from the repo root, with the `forum-tb` environment:

    export OPENAI_API_KEY=...

    python scripts/agent/shap_agent.py \
        --predictions results/predictions/ERR040120_predictions.json \
        --backend openai --model gpt-4o --trace \
        > results/agents/ERR040120_report.txt \
        2> results/agents/ERR040120_trace.txt

Use that environment's python explicitly. A bare `python` picks up whatever
conda env is active and fails at import if it lacks joblib. Environment and
pinned versions: `scripts/agent/Overall_scripts_fastq_train_predict_agent_architecture.md`.
