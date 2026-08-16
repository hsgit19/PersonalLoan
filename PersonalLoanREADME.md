# Personal Loan Campaign — Decision Tree Classification

## Business Problem
Thera Bank ran a personal loan campaign that achieved only a 9.6% conversion 
rate (480 of 5000 customers accepted). This project builds a classification 
model to predict which customers are likely to accept a personal loan offer, 
helping the bank target future campaigns more precisely and reduce wasted 
marketing spend.

## Dataset
5000 customer records, 14 original attributes covering demographics, 
financial behavior, and existing banking relationships. Target variable: 
`Personal_Loan` (binary, 9.6% positive class — imbalanced).

## Data Cleaning & EDA
- Verified no true nulls, no duplicate rows, and no disguised missing-value 
  placeholders (checked every column's min/max for plausibility)
- Dropped `ZIPCode` — 467 unique values across 5000 rows (~10 customers per 
  ZIP on average), too sparse for a decision tree to learn reliable patterns; 
  the raw numeric value also carries no real ordering
- Dropped `ID` — pure row identifier, no predictive value
- Fixed data entry errors in `Experience` (52 rows with implausible negative 
  values, all for customers aged 23–29) by taking the absolute value; verified 
  the fix produced believable "career start age" values (20–30, tightly 
  clustered around 25) via a derived sanity-check column
- Investigated redundancy between `Age` and `Experience` (Spearman correlation 
  = 0.994); `calculated_start_age` analysis suggested `Experience` is likely 
  derived from `Age` rather than independently collected. Initially retained 
  both for modeling since decision trees tolerate multicollinearity; the 
  question was resolved empirically via feature importance (see below)

## Handling Class Imbalance
Two independent strategies were compared:
1. **`class_weight='balanced'`** — penalizes misclassifying the minority class 
   during training, no data transformation needed
2. **SMOTE** — synthetic minority oversampling, applied only to training data 
   to avoid leaking synthetic examples into the test set

## Modeling & Evaluation
Evaluated using `classification_report` and confusion matrix rather than raw 
accuracy, given the imbalance. Both untuned models showed signs of overfitting 
(perfect or near-perfect training accuracy). `GridSearchCV` (5-fold CV, 
optimized for F1 on the minority class) was used to tune `max_depth`, 
`min_samples_split`, and `min_samples_leaf` for both approaches.

### Model comparison

| Model | Precision (1) | Recall (1) | F1 (1) | Train acc | Test acc | Gap |
|---|---|---|---|---|---|---|
| Untuned, class_weight | 0.89 | 0.97 | 0.93 | 1.000 | 0.985 | 0.015 |
| Untuned, SMOTE | 0.77 | 0.96 | 0.85 | 1.000 | 0.968 | 0.032 |
| Tuned, class_weight | 0.66 | 0.99 | 0.79 | 0.962 | 0.950 | 0.012 |
| **Tuned, no class_weight** | **0.93** | **0.93** | **0.93** | **0.983** | **0.986** | **-0.003** |
| Tuned, SMOTE | 0.73 | 0.97 | 0.83 | 0.970 | 0.963 | 0.007 |

**Selected model: Tuned, no class_weight.** It achieves the best balance of 
precision and recall (0.93/0.93) among all five variants, and shows no 
evidence of overfitting — test accuracy (0.986) is essentially identical to, 
even marginally exceeding, training accuracy (0.983). Combining `class_weight` 
with F1-optimized tuning over-corrected toward recall at a steep cost to 
precision (49 false positives vs. 7), so class weighting was dropped once 
hyperparameter tuning was already handling the imbalance directly.

## Feature Importance
```
Income                0.469
Education              0.341
Family                 0.144
CCAvg                  0.043
Age                    0.003
Experience             0.000
(all others)           0.000
```

`Income` and `Education` alone drive over 80% of the model's decisions. 
`Experience` scored exactly zero — dropping it entirely from the feature set 
produced identical performance across every metric (precision, recall, F1, 
train/test accuracy), confirming it was redundant with `Age` and safe to remove.

**Note on `CD_Account`:** despite scoring zero importance in this tree, 
customers with a CD account accept loans at 46.4% vs. 7.2% for those without 
— a genuinely strong standalone signal. Its zero importance reflects that 
`Income`/`Education` already captured most of the same separating power in 
earlier splits, not that the relationship doesn't exist. Worth surfacing to 
the bank as a targeting signal independent of what this particular tree used.

## Business Recommendations
1. Deploy the tuned, no-class_weight decision tree — balanced precision/recall 
   means the bank neither misses too many real prospects nor over-spends on 
   false leads.
2. Prioritize customers with high `Income`, higher `Education` levels, and 
   larger `Family` size in future campaign targeting.
3. Customers holding a `CD_Account` convert at 6x the rate of those without — 
   a strong, simple targeting signal worth using directly in campaign selection, 
   even though the tree itself didn't rely on it.

## Future Work
- Compare against Random Forest / Gradient Boosting for a raw accuracy ceiling
- Cost-sensitive threshold tuning (adjust the probability cutoff rather than 
  relying on the default 0.5) to explicitly trade off precision vs. recall 
  based on the bank's actual cost of a false positive vs. false negative

## Tech Stack
Python, pandas, scikit-learn, imbalanced-learn (SMOTE), matplotlib/seaborn
