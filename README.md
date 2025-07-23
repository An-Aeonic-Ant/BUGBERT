# BugBERT: NLP for Bug Prioritization

**Automating the prioritization of bug reports using DistilBERT, feature engineering, and machine learning.**

![BugBERT Banner](#) <!-- Replace with actual image link -->

## Overview

Manual bug triaging is labor-intensive, subjective, and doesn't scale with the volume of bug reports in large software systems. **BugBERT** is a hybrid ML-DL pipeline that leverages state-of-the-art NLP (DistilBERT), structured feature engineering, and robust classifiers (XGBoost) to automate bug priority classification with promising results.

---

## Dataset

- **Total rows**: ~53,000
- **Independent features**:
  - `Title` (short text)
  - `Description` (long text)
  - `Component` (categorical, high cardinality)
  - `Status`, `Resolution` (categorical)
- **Target variable**: `Priority` (5 classes, heavily imbalanced)

> `Issue_id` was dropped as it carried no useful signal.

---

## Methodology

### 1. Language Modeling with DistilBERT

- Extract sentence-level embeddings using `[CLS]` token
- Input: Concatenated `Component`, `Title`, and `Description`
- Fine-tuned using **class-weighted loss** to handle imbalance

### 2. Categorical Feature Engineering

- **Frequency Encoding**: Based on category occurrence rates
- **Target Encoding**: Based on mode priority class per category (with care to avoid leakage)

### 3. Dimensionality Reduction & Clustering

- **PCA**: Reduced 768-dim embeddings to lower dimensions
- **K-Means Clustering**: Added cluster IDs as new features

### 4. Handling Class Imbalance

- **Sample Weights**: Upweight minority classes during training
- **SMOTE-NC**: Synthetic oversampling for numerical + categorical features

### 5. Final Classification

- **XGBoost**: Trained on combined feature space
- **Hyperparameter tuning**: Done with **Optuna**

---

## Results

| Priority | Precision | Recall | F1-Score | Support |
|----------|-----------|--------|----------|---------|
| 0        | 0.46      | 0.50   | 0.48     | 2236    |
| 1        | 0.36      | 0.34   | 0.35     | 2353    |
| 2        | 0.79      | 0.74   | 0.76     | 7789    |
| 3        | 0.21      | 0.33   | 0.26     | 572     |
| 4        | 0.20      | 0.25   | 0.22     | 299     |

- **Validation Accuracy**: 63%
- **Validation Macro F1**: 0.41
- **Test Macro F1**: 0.42

> Shows strong performance in dominant classes while minority classes benefit from balancing techniques.

---

---

## 📷 Visualizations

<!-- Replace '#' with actual image URLs or GitHub relative paths -->

### Pipeline Architecture  
![Pipeline Diagram](#)

### Dimensionality Reduction  
![PCA Explained Variance](#)

### Priority Class Distribution  
![Class Imbalance Chart](#)

---

## Technologies Used

- Python, Jupyter Notebooks
- HuggingFace Transformers (DistilBERT)
- Scikit-learn, Imbalanced-learn, XGBoost, Optuna
- SMOTE-NC, PCA, KMeans

---

## Acknowledgements

- **Anant Maheshwari**, ABV-IIITM Gwalior  
- **Dr. Anjali**, ABV-IIITM Gwalior (Supervisor)

---

## Contact

For questions or collaboration opportunities, feel free to reach out:

anant200519@gmail.com

---




