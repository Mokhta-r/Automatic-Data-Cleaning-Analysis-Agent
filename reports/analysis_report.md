# 📊 Data Analysis Report

## 1️⃣ Dataset Overview
- Shape: **(20, 4)**
- Columns: id, age, income, city

## 2️⃣ Cleaning Summary
- Shape before: (20, 4)
- Shape after: (20, 4)
- Dropped columns: []
- Duplicates removed: 0

### Imputations
- **age** → median (imputed 5 values)
- **income** → median (imputed 6 values)
- **city** → mode (imputed 6 values)

## 3️⃣ Descriptive Statistics
### id
- count: 20.0
- mean: 10.5
- std: 5.916079783099616
- min: 1.0
- 25%: 5.75
- 50%: 10.5
- 75%: 15.25
- max: 20.0
### age
- count: 20.0
- mean: 34.05
- std: 5.145002685283939
- min: 25.0
- 25%: 31.5
- 50%: 34.0
- 75%: 36.25
- max: 44.0
### income
- count: 20.0
- mean: 4005.0
- std: 430.0856707802834
- min: 3000.0
- 25%: 3875.0
- 50%: 4050.0
- 75%: 4225.0
- max: 4700.0

## 4️⃣ Correlations
### Strong correlations:
- **id** ↔ **income** → 0.82
- **age** ↔ **id** → 0.88
- **age** ↔ **income** → 0.87

## 5️⃣ Distribution Summaries
### id
- min: 1.0
- max: 20.0
- mean: 10.5
- median: 10.5
- skewness: 0.0
- q1: 5.75
- q3: 15.25
### age
- min: 25.0
- max: 44.0
- mean: 34.05
- median: 34.0
- skewness: 0.10715522387337376
- q1: 31.5
- q3: 36.25
### income
- min: 3000.0
- max: 4700.0
- mean: 4005.0
- median: 4050.0
- skewness: -0.7188934095275396
- q1: 3875.0
- q3: 4225.0

## 6️⃣ Insights
- Strong correlation (0.82) between **id** and **income**.
- Strong correlation (0.88) between **age** and **id**.
- Strong correlation (0.87) between **age** and **income**.
- Column **age** had 5 missing values imputed.
- Column **income** had 6 missing values imputed.
- Column **city** had 6 missing values imputed.
- Column **age** contains 1 outliers.
- Column **income** contains 2 outliers.

## 7️⃣ Warnings
- ⚠️ High number of imputations for **income** (6).
- ⚠️ High number of imputations for **city** (6).
- ⚠️ Outliers detected but not removed.