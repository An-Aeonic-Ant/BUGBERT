# BugBERT: NLP for Bug Prioritization

**Automating the prioritization of bug reports using DistilBERT, feature engineering, and machine learning.**

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

<img width="530" height="352" alt="image" src="https://github.com/user-attachments/assets/751ea713-9c9d-4e8a-b5cf-d992338aae8b" />


> Shows strong performance in dominant classes while minority classes benefit from balancing techniques.

---

---

## 📷 Visualizations

### Priority Class Distribution  
<img width="784" height="840" alt="image" src="https://github.com/user-attachments/assets/1f488151-23c8-4e8b-bbfe-5abb12bf2a1f" />

### Dimensionality Reduction  
<img width="806" height="486" alt="image" src="https://github.com/user-attachments/assets/d63b1d18-2f14-42f6-bb13-6298ba5a79c1" />

### Pipeline Architecture  
<img width="1353" height="523" alt="image" src="https://github.com/user-attachments/assets/9b109e2e-bb8e-4e60-a8a1-dbb50cd40ebd" />



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




