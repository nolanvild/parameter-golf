# MLP Nonlinearity Sweep — Results Summary

**Config:** 3-layer transformer, dim=128, 4 heads / 2 kv-heads, 500-step budget (2 min wallclock cap),
batch 4096 tokens, seq_len=256. Metric: `final_int8_zlib_roundtrip_exact` (int8-quantised model,
zlib-compressed roundtrip eval on validation set).

Seeds: 42, 123, 1337 — all stats use sample std (n−1).

---

## Per-run results

| Power | Nonlinearity | Seed | val_loss     | val_bpb    |
|-------|--------------|------|--------------|------------|
| 1     | x (linear)   | 42   | 4.40243349   | 2.60298490 |
| 1     | x (linear)   | 123  | 4.41038361   | 2.60768550 |
| 1     | x (linear)   | 1337 | 4.39137764   | 2.59644802 |
| 2     | x²           | 42   | 4.42508912   | 2.61638029 |
| 2     | x²           | 123  | 4.39156388   | 2.59655814 |
| 2     | x²           | 1337 | 4.41671303   | 2.61142783 |
| 3     | x³           | 42   | 4.92961843   | 2.91468852 |
| 3     | x³           | 123  | 4.94636972   | 2.92459290 |
| 3     | x³           | 1337 | 4.45692760   | 2.63520512 |
| 4     | x⁴           | 42   | 5.01119875   | 2.96292374 |
| 4     | x⁴           | 123  | 5.01226449   | 2.96355387 |
| 4     | x⁴           | 1337 | 4.99858513   | 2.95546581 |
| 5     | x⁵           | 42   | 5.11442078   | 3.02395485 |
| 5     | x⁵           | 123  | 5.11533808   | 3.02449721 |
| 5     | x⁵           | 1337 | 5.08754288   | 3.00806301 |

---

## Aggregate statistics (mean ± std across 3 seeds)

| Power | Nonlinearity | val_loss mean | val_loss std | val_bpb mean | val_bpb std |
|-------|--------------|---------------|--------------|--------------|-------------|
| 1     | x (linear)   | 4.40139825    | 0.00954518   | 2.60237281   | 0.00564369  |
| 2     | x²           | 4.41112201    | 0.01744792   | 2.60812209   | 0.01031627  |
| 3     | x³           | 4.77763858    | 0.27787012   | 2.82482885   | 0.16429362  |
| 4     | x⁴           | 5.00734946    | 0.00760881   | 2.96064781   | 0.00449879  |
| 5     | x⁵           | 5.10576725    | 0.01578943   | 3.01883836   | 0.00933566  |

---

## Notes

- **x (linear) and x²** perform comparably and best overall (~2.60–2.61 bpb).
- **x³** has a high-variance outlier: seed=1337 lands at 2.635 bpb while seeds 42 and 123
  land at ~2.91–2.92 bpb, suggesting x³ is sensitive to initialisation.
- **x⁴ and x⁵** converge consistently but to worse loss (~2.96 and ~3.02 bpb respectively),
  likely because high-power nonlinearities saturate or produce very small gradients early in training.
- All runs hit the 2-minute wallclock cap (stopped at step ~184–190 out of 500), so these
  results reflect early-training dynamics rather than converged performance.
