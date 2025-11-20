# Automatic-Data-Cleaning# Automatic Data Cleaning & Analysis Agent

## 🎯 Objective

This project implements an **Automatic Data Cleaning & Analysis Agent** that:
- Loads raw **CSV/Excel** datasets
- Automatically **profiles**, **cleans**, and **analyzes** the data
- Exports:
  - A **cleaned dataset** ready for modeling
  - A **Markdown/PDF report** summarizing structure, issues, and insights

## 🧩 Problem

Data scientists and analysts spend a lot of time:
- Inspecting dataset structure (shape, columns, dtypes)
- Handling missing values and duplicates
- Fixing inconsistent types
- Detecting simple outliers
- Writing the same boilerplate code again and again

This preprocessing phase is:
- Time-consuming
- Error-prone
- Repeated in every new project

## ✅ Proposed Solution

Build an **agent-style system** that:

1. **Loads** any CSV/Excel file
2. **Analyzes** the structure:
   - Shape, columns, data types
   - Missing value report
   - Duplicates
3. **Cleans** the data:
   - Handles missing values (per column type)
   - Detects & removes duplicates
   - Fixes inconsistent types
   - Detects basic outliers
4. **Analyzes** the cleaned dataset:
   - Descriptive statistics
   - Correlations
   - Distribution summaries
   - Insights + warnings
5. **Exports**:
   - `cleaned_dataset.csv` (or .xlsx)
   - `analysis_report.md` (and optionally PDF)

## 🏗️ High-Level Architecture

```text
User dataset (CSV/Excel)
          │
          ▼
  Data Loader Tool
          │
          ▼
  Data Profiler Tool
          │
          ▼
   Data Cleaner Tool
          │
          ▼
  Analysis & Insights Tool
          │
          ▼
 Report & Export Tool
   ├─ cleaned_dataset.csv
   └─ analysis_report.md/.pdf
