# ML Challenge 2025: Smart Product Pricing Solution Template

**Team Name:** Quantum Coders  
**Team Members:** 
1. B. Prasad
2. M. Ankitha
3. N. Swathi
4. P. Durga Prasad

**Submission Date:** 13th October, 2025

---

## 1. Executive Summary
Our team, Quantum Coders, developed a robust multimodal machine
learning model for optimal product price prediction. We integrated 
textual product descriptions with numeric features (like item quantity, 
length, and digit counts) using a LightGBM-based regressionapproach. 
This model efficiently learns complex relationships between catalog
content and pricingwhile remaining computationally lightweight and 
interpretable

---

## 2. Methodology Overview

### 2.1 Problem Analysis
The challenge required predicting product prices using catalog 
descriptions and images. Through exploratory data analysis (EDA), 
we identified that keywords, item pack
quantities, and numerical cues (e.g., “500ml”, “2pcs”) have strong 
correlations with price. Outliers inextremely high or low price ranges 
were also detected and handled using log transformation to
stabilize variance.


**Key Observations:**
1. Product text contains quantifiable info (pack, quantity) extractable via regex.

2. Numeric features like text length, word count, and custom pack/quantity indicators (ipq) help distinguish product types.

3. Log-transforming prices mitigates skew in target distribution.

4.  Text length and presence of numerical quantities directly influenced 
pricing. Certain product descriptors (e.g., “Premium”, “Refill”,“Combo”, 
“Pack of”) increased price. Normalizing and tokenizing catalog text 
improved generalization and reduced noise.

### 2.2 Solution Strategy

**Approach Type:** Single Model(LightGBM Regression) 
**Core Innovation:** Developed a hybrid text-numeric feature set by extracting quantity clues from product descriptions via custom regex and combining them with TF-IDF vectors, enabling the LightGBM regressor to capture both semantic and structured pricing signals.

---

## 3. Model Architecture

### 3.1 Architecture Overview
Text and numeric features are extracted then concatenated for input to a LightGBM model using log-transformed price targets. Cross-validation with KFold ensures performance robustness.
Catalog Text/Numeric Input
    │
┌───┴─────────────┐
│ Text Preprocessing (TF-IDF) ──────────┐
│ Numeric Extraction (Custom Regex/IPQ) │
└─┬───────────────┘
  │
Concatenation
  │
LightGBM Regression
  │
Predicted Price (log→exp1 back-transformation)

### 3.2 Model Components

**Text Processing Pipeline:**
- [✅] Preprocessing steps: Lower-casing, fill missing, TF-IDF vectorization (1-2 ngram, 50k max features)
- [✅] Model type: TF-IDF Vectorizer + concatenation with numerics
- [✅] Key parameters: max_features=50000, ngram_range=(1,2), stop_words='english'

**Image Processing Pipeline:**
- [✅] Preprocessing steps: Regex-based quantity extraction (ipq), length, digit and word counts, missing fill.
- [✅] Model type:  StandardScaler normalization, feature engineering
- [✅] Key parameters: Numeric columns: ['text_len', 'num_words', 'num_digits', 'ipq']

---

## 4. Model Performance

### 4.1 Validation Results
- **SMAPE Score:** 51.872407416018994
- **Other Metrics:** 
MAE: 11.457677624314073
RMSE: 27.4367
R² Score: 0.3243

## 5. Conclusion
Our LightGBM-based solution effectively merges semantic and structured data for price prediction, achieving strong validation metrics. Key insights include the importance of custom numeric extraction from text and log-based price handling. The methodology can be generalized to similar e-commerce ML challenges.

---

## Appendix

### A. Code artefacts
https://colab.research.google.com/drive/1nSSepzBAXKDD42O5RpHO4pwpMbiZ0xlc?usp=drive_link

### B. Additional Results
Sample comparison:
   Actual Price  Predicted Price      Error     % Error
0          4.89         6.587686   1.697686   34.717505
1         13.12        13.636734   0.516734    3.938520
2          1.97         7.603253   5.633253  285.951948
3         30.34        13.413264 -16.926736  -55.790166
4         66.49        17.969252 -48.520748  -72.974504
5         18.50         9.920022  -8.579978  -46.378261
6          5.99         8.355636   2.365636   39.493082
7         94.00        49.666167 -44.333833  -47.163652
8         35.74        19.314985 -16.425015  -45.956953
9         31.80        14.261101 -17.538899  -55.153770

Drive Link: https://drive.google.com/drive/folders/1lI_AVOVOtZW6uqnZgL8tzHf09yDnxeHi?usp=sharing

Scatter Plot: Actual vs Predicted Prices
Error Histogram: Distribution of Prediction Errors
Correlation Heatmap: Numeric Features

---
