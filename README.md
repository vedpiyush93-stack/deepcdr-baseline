# Simplified DeepCDR Baseline

> A pared-down DeepCDR, used as the reference arm in the drug-response stacking experiments.

---

Forked from the original DeepCDR and reduced to the components the stacking study needed, so that
baseline and stacker share one data path — without which any accuracy comparison between them would be
confounded by preprocessing differences.

## Repository layout

```
SimplerDeepCDR/                              36 files
data/                                        8 files
```

`SimplerDeepCDR/` holds the model and training notebooks · `data/` the small feature tables.

## Notebooks

**31 notebooks**, committed with their outputs intact so every figure and result table renders on GitHub without re-running anything.

## About this repository

Research code from my doctoral work at the University of Nebraska–Lincoln. Trained model checkpoints and bulk datasets are excluded from version control; the notebooks regenerate them. Previously hosted at `github.com/Ved-Piyush/DeepCDR_SimpleCDR`.

---

**Ved Piyush, PhD** · Statistics, University of Nebraska–Lincoln  
[vedpiyush93@gmail.com](mailto:vedpiyush93@gmail.com) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=657rVYAAAAAJ) · [LinkedIn](https://www.linkedin.com/in/ved-piyush)