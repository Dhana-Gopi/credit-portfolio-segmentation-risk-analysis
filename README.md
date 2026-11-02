# Credit Portfolio Segmentation & Risk Pattern Analysis

## Project Overview

An end-to-end machine learning project that analyzes historical credit portfolio data, identifies borrower segments, and explores financial risk patterns using unsupervised learning.

The project includes exploratory data analysis, feature engineering, clustering model comparisons, segment-level business insights, and an interactive Streamlit dashboard.

## Project Objectives

- Analyze borrower financial and credit characteristics.
- Identify meaningful borrower groups using clustering.
- Compare historical default patterns across segments.
- Evaluate K-Means, Hierarchical Clustering, and DBSCAN.
- Build an interactive dashboard for portfolio analysis.
- Assign new borrower records to existing K-Means segments.

## Dataset

- Total records: 1,347,681
- Original features: 15
- Final model features: 24
- Historical outcome: `Default` (1 = Default, 0 = Non-default)

## Technologies Used

Python, Pandas, NumPy, Scikit-learn, Plotly, Streamlit, Joblib and Jupyter Notebook.

## Machine Learning Approach

**Data preprocessing:** Data validation, outlier treatment, income transformation, employment feature engineering, StandardScaler and OneHotEncoder.

**Clustering:** K-Means, Hierarchical Clustering and DBSCAN.

**Model evaluation:** Silhouette scores, cluster comparisons and business interpretation.

**Final model:** K-Means with five borrower segments.

## Identified Borrower Segments

1. Higher-FICO Mortgage-Oriented
2. Shorter-Employment Renters
3. Higher-Income Larger-Loan
4. Higher Debt-Burden
5. Longer-Employment Lower-DTI

## Key Business Insights

- Higher Debt-Burden borrowers showed the highest historical segment default rate, approximately 28.99%.
- Higher-Income Larger-Loan borrowers represented approximately 42.21% of original loan principal in the test portfolio.
- Borrower segments showed different historical financial and default patterns.

## Streamlit Dashboard

The dashboard contains five sections:

1. Executive Overview
2. Segment Intelligence
3. Model Evaluation
4. Anomaly Analysis
5. Borrower Segmentation

The Borrower Segmentation page accepts borrower financial details and assigns a segment using the saved preprocessing pipeline and K-Means model.

## How to Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the dashboard:

```bash
streamlit run app.py
```

Keep the saved model files in the `models/` directory and the scorecard CSV in `project_outputs/`.

## Limitations

This project is for exploratory portfolio analytics and educational research. K-Means does not predict an individual borrower's probability of default. Historical segment default rates should not be used as individual risk predictions or automated lending decisions.

## Future Improvements

- Add supervised default prediction with appropriate evaluation.
- Monitor segment stability across time.
- Extend borrower-level dashboard filtering.
- Improve model explainability and data validation.
- Deploy the dashboard for demonstration.

## Project Type

Machine Learning | Unsupervised Learning | Financial Analytics | Streamlit Dashboard
