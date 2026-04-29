# 🚓 Chicago Crime Analysis & Clustering Project

## 📌 Overview

This project presents an end-to-end data science pipeline for analyzing crime patterns in Chicago using machine learning techniques. It includes data preprocessing, feature engineering, clustering, dimensionality reduction, MLflow experiment tracking, and deployment via a Streamlit dashboard.

---

## 🎯 Objectives

* Analyze crime distribution across time and geography
* Identify hidden patterns using clustering algorithms
* Reduce dimensionality for visualization
* Track experiments using MLflow
* Build an interactive dashboard for insights

---

## 📊 Dataset Information

**Source:** Chicago Data Portal – Crimes 2001 to Present (Public Dataset)

* Dataset Scale: 7.8 million records (2001–2025)
* Sample Used: ~500,000 records
* Features: 22 variables
* Crime Categories: 33 types
* Coverage: Chicago districts and wards

**Time Range Used in Project:**

* Start Date: 2023-04-09
* End Date: 2025-03-15

---

## 📂 Dataset Access

Due to GitHub size limitations, large datasets are hosted on Google Drive:

* chicago_crime_with_features.csv → https://drive.google.com/file/d/1WWs8zcQtV6AdKJrISgBk228qvI_pgarJ/view?usp=sharing
* clustering_results.csv → https://drive.google.com/file/d/1X7ckBqZO304JiBFpsFJkiukT9g1zFwiA/view?usp=sharing
* crime_dbscan_clustered.csv → https://drive.google.com/file/d/1nFJu7Sl_iSLACyai22BfE66LoG9-YJrw/view?usp=sharing

---

## ⚙️ How to Run the Project

### 🔹 Option 1: Quick Start (Recommended)

1. Download datasets from the links above
2. Place them inside the `Data/` folder
3. Run the app:

   ```bash
   streamlit run app.py
   ```

---

### 🔹 Option 2: Reproduce Full Pipeline

Run notebooks in order:

1. `data_cleaning.ipynb`
2. `eda.ipynb`
3. `feature_eng.ipynb`
4. `temporal_cluster.ipynb`
5. `mlflow_tracking.ipynb`
6. Run `app.py`

---

## 🧠 Machine Learning Techniques

### Clustering Models

* KMeans Clustering
* DBSCAN
* Hierarchical Clustering

### Dimensionality Reduction

* PCA (Principal Component Analysis)
* t-SNE

---

## 📈 MLflow Tracking

MLflow is used to:

* Track experiments
* Compare model performance
* Store parameters and metrics

---

## 🖥️ Streamlit Application Features

* Interactive crime analysis dashboard
* Cluster visualization
* PCA and t-SNE visualizations
* User-friendly interface

---

## 📁 Project Structure

```
CHICAGO_PATROLQ_PROJECT/
├── Data/
├── Data_prep/
├── mlruns/
├── app.py
├── requirements.txt
├── model_metadata.json
├── Project_documentation.pdf
└── README.md
```

---

## ⚠️ Important Notes

* Large datasets are not included in the repository due to size limits
* Please download required files before running the app
* MLflow tracking data is included for experiment reference

---

## 🚀 Deployment

The application is designed to be deployed on Streamlit Cloud with automatic updates via GitHub integration.

---

## 📌 Conclusion

This project demonstrates a complete data science workflow from raw data processing to model deployment, combining machine learning, experiment tracking, and interactive visualization.

---

## 👤 Author

**Rama Sekar**

---
