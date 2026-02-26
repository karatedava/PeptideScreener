## Solubility Predictor — README  

---

### 1. Overview
A single-sequence CLI tool (`predict_solubility.py`) that returns the probability of a peptide being **soluble** (0 – 100 %).

---

### 2. Installation
```bash
conda create -n solpred python=3.9 -y
conda activate solpred
python -m pip install localcider pandas joblib tqdm   torch==2.2.2+cpu -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install git+https://github.com/idptools/sparrow.git
python -m pip install GPy==1.10.0
```

---

### 3. Inference on a single peptide
```bash
python predict_solubility.py "KKQFFFEYKLMLSMAKFESAM"
# → Predicted solubility: 95.7 %
```

---

### 4. Performance

254 Peptides

| Metric | Mean ± SD |
|--------|-----------|
| **Accuracy** | **0.831 ± 0.058** |
| **Matthews CC** | **0.608 ± 0.131** |
| **ROC AUC** | **0.892 ± 0.055** |

Hold-out test (51 sequences):

* Accuracy   = 0.824  
* MCC        = 0.661  
* ROC AUC   = 0.913  

Confusion matrix  

```
           Pred Insol | Pred Sol
Actual Insol     15   |     1
Actual Sol        8   |    27
```

---
