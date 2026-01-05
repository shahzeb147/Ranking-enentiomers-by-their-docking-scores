# Ranking Enantiomers by Their Docking Scores

This repository contains code and experiments for **ranking enantiomer pairs based on molecular docking scores**. This work is build on top of our project on molecule featurization for drug discovery. 

---

## Dataset Description

Each enantiomer may have multiple 3D conformers. All conformers belonging to the same enantiomer are assigned the **same docking score**, corresponding to the **best (lowest) docking score** observed for that enantiomer. This ensures that docking score is modeled at the **enantiomer level**.

### Dataset Statistics
- **335K conformers**
- **69K enantiomers** (≈ 34.5K enantiomer pairs)
- Each enantiomer has multiple conformers sharing the same target value

### Data Splitting
- **70% training**
- **15% validation**
- **15% test**

Enantiomer pairs are always kept within the **same split** to avoid data leakage.

---

## Training Setup

- Models are trained using **Mean Squared Error (MSE)** loss on docking scores
- During training, one of the following strategies is used:
  - Randomly sample **one conformer per enantiomer** in each batch
  - Use a **fixed pre selected conformer** per enantiomer


---

## Evaluation

- Evaluation is based on **ranking accuracy** between enantiomer pairs
- Predicted docking scores are **averaged across conformers**
- The averaged scores are compared between enantiomers in the **test set**
- The goal is to correctly identify which enantiomer has the better (lower) docking score

---

## Repository Structure

```text
Ranking-enentiomers-by-their-docking-scores/
├── src/
│   ├── atom_encoder.py        # Single atom properties and feature switches
│   ├── truncated_views.py     # Truncated views with pairwise information
│   └── weighted_views.py      # Original Weighted Views featurization method
│
├── notebooks/
│   ├── full_split.ipynb
│   │   └─ Enantiomer ranking using neural networks only
│   ├── fullsplit_singleproperties.ipynb
│   │   └─ Ranking with additional single atom properties
│   ├── data_information.ipynb
│   │   └─ Dataset analysis and statistics
│   └── chirality_transformer_coulomb.ipynb
│       └─ Transformer based model with Coulomb and chirality aware features
│
└── README.md
```


