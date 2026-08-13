# Simplified DeepCDR Baseline

A pared-down DeepCDR used as the baseline arm in the drug-response stacking experiments.

## Context

Forked from the original DeepCDR and reduced to the pieces the stacking study needed, so baseline and stacker share a data path.

## Repository layout

```
SimplerDeepCDR/                                36 files
data/                                          8 files
```

`SimplerDeepCDR/` holds the model and training notebooks; `data/` the small feature tables.

## Notebooks

31 notebooks, run with their outputs preserved so the results are readable without re-running anything.

## Notes on this repository

Notebook outputs are kept intact, so figures and result tables render directly on GitHub. Genomic feature matrices and trained weights are omitted.

Originally developed during my PhD at the University of Nebraska–Lincoln and previously hosted at `github.com/Ved-Piyush/DeepCDR_SimpleCDR`.

---

**Ved Piyush, PhD** — [vedpiyush93@gmail.com](mailto:vedpiyush93@gmail.com) · [LinkedIn](https://www.linkedin.com/in/ved-piyush) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=657rVYAAAAAJ)