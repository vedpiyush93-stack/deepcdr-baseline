# A Simplified Drug-Response Model, with Dropout Intervals

> The comparison baseline: a lighter architecture that produces uncertainty by Monte Carlo Dropout.

---

**Dropout** randomly switches off parts of a neural network during training to prevent it from memorising.
**Monte Carlo Dropout** is a trick: leave dropout switched on at prediction time too, run the same input through
many times, and read the spread of answers as an uncertainty estimate. It is the standard, easy baseline that any
new uncertainty method has to beat.

This repository holds a pared-down version of the drug-response architecture together with that baseline, so the
comparison against the ensemble Kalman filter method is like-for-like on identical data.

## What the code does

`SimpleCDRGCN_Dropout_Intervals` produces Monte Carlo Dropout intervals and computes **coverage**. Other
notebooks vary the pieces that matter for a fair comparison:

- dropout kept active in training only, versus in training *and* prediction
  (`..._active_only_train_not_pred` versus `..._active_both`);
- a **no-leakage** variant ensuring no cell line appears in both training and test
  (`..._no_leakage`), and a transfer-learning version of it;
- ten-fold cross-validation built from held-out samples;
- feature construction from drug structure and cell-line molecular data, with and without normalisation.

Uses TensorFlow Probability for the probabilistic layers and RDKit/DeepChem to turn drug structures into graphs.

## Repository layout

`SimplerDeepCDR/` holds the notebooks behind the reported results; `SimplerDeepCDR/Dev_Scripts/` holds earlier exploratory versions, including alternative inputs such as 1-D convolutions over chemical strings.

## Running it

1. `SimplerDeepCDR_Drug_Cell_Line_Features_Create_Feature_Mats.ipynb` builds the feature matrices.
2. `SimplerCDR_Exact_Network_more_dropout_no_leakage.ipynb` trains without train/test leakage.
3. `SimpleCDRGCN_Dropout_Intervals.ipynb` produces the Monte Carlo Dropout intervals and coverage.

## Notes

Notebook outputs are committed, so the figures and result tables render on GitHub without running anything. Molecular feature matrices and trained weights are not committed; the feature notebooks rebuild them.

Research code from my doctoral work at the University of Nebraska–Lincoln (31 notebooks). Previously hosted at `github.com/Ved-Piyush/DeepCDR_SimpleCDR`.

---

**Ved Piyush, PhD** · Statistics, University of Nebraska–Lincoln  
[vedpiyush93@gmail.com](mailto:vedpiyush93@gmail.com) · [Google Scholar](https://scholar.google.com/citations?hl=en&user=657rVYAAAAAJ) · [Website](https://vedpiyush93-stack.github.io)