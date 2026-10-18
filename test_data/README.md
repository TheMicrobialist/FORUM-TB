# Test data

Two isolates, both with complete four-drug reference output.

| File | Isolate | SNPs | Profile |
|---|---|---|---|
| `ERR040120.filtered.vcf.gz` | MDR-TB | 944 | Resistant to all four drugs |
| `ERR038266.vcf.gz` | pan-susceptible | 1,659 | Susceptible to all four |

Used by `scripts/vcf_to_prediction.py`, `scripts/vcf_to_waterfall.py` and the
dashboard. `ERR038266.vcf.gz` is also kept in `example_output/`.

Needs numpy >= 2.0 and scikit-learn 1.6.1 (see #34, #35) and the trained models
(see `models/README.md`).

**1. Check the data loads.** No models or API key needed:

```bash
python3 scripts/agent/test_shap_agent_tools.py
```

Expect `37 passed, 0 failed`.

**2. Get the models.** 244 MB, not in git — snippet in `models/README.md`.

**3. Run it:**

```bash
python3 scripts/vcf_to_prediction.py \
    --vcf test_data/ERR040120.filtered.vcf.gz \
    --all-drugs
```

**4. Check your output matches:**

| Drug | ERR040120 | P | ERR038266 | P |
|---|---|---|---|---|
| RIFAMPICIN | Resistant | 1.0000 | Susceptible | 0.0000 |
| ISONIAZID | Resistant | 1.0000 | Susceptible | 0.0346 |
| ETHAMBUTOL | Resistant | 0.7334 | Susceptible | 0.0000 |
| PYRAZINAMIDE | Resistant | 0.8508 | Susceptible | 0.0000 |

Both match the ground-truth labels 4/4. Note both isolates are in the training
split, so these are in-sample.

WHO-graded variants found in ERR040120: `rpoB_p.Ser450Leu`,
`katG_p.Ser315Thr`, `embB_p.Gly406Asp`, all graded `1) Assoc w R`. None in
ERR038266, which is expected for a susceptible isolate — there the
explanations turn on resistance sites carrying the reference base.

Top rifampicin feature for ERR040120: `pos_761155` (*rpoB* S450L), SHAP
+0.280542. Reference output: `results/predictions/<SAMPLE>_predictions.json`.

**5. Explain the result** (optional, needs an API key):

```bash
python3 scripts/agent/shap_agent.py \
    --predictions results/predictions/ERR040120_predictions.json --trace
```
