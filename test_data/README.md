# Test data

`ERR040120.filtered.vcf.gz` — MDR-TB isolate, 944 SNPs. The example VCF used by
`scripts/vcf_to_prediction.py`, `scripts/vcf_to_waterfall.py` and the dashboard.

Needs numpy >= 2.0 (see #34) and the trained models (see `models/README.md`).

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

| Drug | Prediction | P(resistant) |
|---|---|---|
| RIFAMPICIN | Resistant | 1.0000 |
| ISONIAZID | Resistant | 1.0000 |
| ETHAMBUTOL | Resistant | 0.7334 |
| PYRAZINAMIDE | Resistant | 0.8508 |

Top rifampicin feature: `pos_761155` (*rpoB* S450L), SHAP +0.280542.
Reference output: `results/predictions/ERR040120_predictions.json`.

**5. Explain the result** (optional, needs an API key):

```bash
python3 scripts/agent/shap_agent.py \
    --predictions results/predictions/ERR040120_predictions.json --trace
```
