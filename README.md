# 🚓 Chicago Crime Intelligence Platform

An interactive data science project that analyzes ~500,000 real-world crime records to uncover **spatial, temporal, and behavioral crime patterns** using clustering and dimensionality reduction techniques.

👉 **Live Application:**
https://chicago-crime-analysis-jatjuq9fu8kdjydgr2qkmh.streamlit.app/

---

## 📸 Application Preview

### 🔹 Dashboard Overview
<img width="1919" height="875" alt="Screenshot 2026-05-03 194329" src="https://github.com/user-attachments/assets/92b67b6b-1f52-49ab-9997-c732ce864a2f" />


### 🔹 Clustering Visualization
<img width="1913" height="866" alt="Screenshot 2026-05-03 194420" src="https://github.com/user-attachments/assets/0530da8c-b6a9-4446-901d-c78c1ec9c226" />


### 🔹 Dimensionality Reduction (PCA / t-SNE)
<img width="1915" height="865" alt="Screenshot 2026-05-03 194539" src="https://github.com/user-attachments/assets/3fbf800e-352f-43ff-b0c7-0e5454e6f6a6" />

---

## 🎯 Objectives

* Analyze crime distribution across **time and geography**
* Identify hidden patterns using **unsupervised learning**
* Reduce high-dimensional data for **visual interpretation**
* Track experiments using **MLflow**
* Build an **interactive dashboard** for real-time insights

---

## 🔍 Key Insights

* High-density crime clusters are concentrated in specific districts
* Evening and late-night hours show peak crime activity
* **DBSCAN** identified dense hotspots missed by K-Means
* Spatial features (location, district) are dominant contributors
* PCA retained **~74% variance** with just 7 components

---

## 📊 Dataset Information

* **Source:** Chicago Data Portal – Crimes 2001 to Present
* **Total Records Available:** 7.8 Million
* **Sample Used:** ~500,000 records
* **Features:** 22 features
* **Crime Categories:** 33 types
* **Coverage:** Chicago districts and wards

### ⏳ Time Range Used

* Start Date: 2023-04-09
* End Date: 2025-03-15

---

## 📂 Dataset Access

Due to GitHub size limits, datasets are hosted externally:

* chicago_crime_with_features.csv
  https://drive.google.com/file/d/1WWs8zcQtV6AdKJrISgBk228qvI_pgarJ/view?usp=sharing

* clustering_results.csv
  https://drive.google.com/file/d/1X7ckBqZO304JiBFpsFJkiukT9g1zFwiA/view?usp=sharing

* crime_dbscan_clustered.csv
  https://drive.google.com/file/d/1nFJu7Sl_iSLACyai22BfE66LoG9-YJrw/view?usp=sharing

---

## 🧠 Machine Learning Techniques

### 🔹 Clustering

* K-Means Clustering
* DBSCAN
* Hierarchical Clustering

### 🔹 Dimensionality Reduction

* PCA (Principal Component Analysis)
* t-SNE

---

## 📈 MLflow Tracking

MLflow is used to:

* Track experiments
* Compare clustering performance
* Store parameters and evaluation metrics

---

## 🖥️ Streamlit Application Features

* 📊 Interactive crime analytics dashboard
* 🗺️ Geographic crime visualization
* ⏰ Temporal pattern analysis
* 🔍 Clustering comparison (KMeans vs DBSCAN)
* 🧠 PCA & t-SNE visualizations
* 🎛️ Dynamic filters (year, district, crime type)

---

## 🛠️ Tech Stack

* Python
* Pandas, NumPy
* Scikit-learn
* MLflow
* Streamlit
* Matplotlib / Seaborn

---

## ⚙️ Run Locally

1. Download datasets from the links above
2. Place them inside the `Data/` folder
3. Run the application:

```bash
streamlit run app.py
```

---

## 📁 Project Structure

```
CHICAGO_PATROLQ_PROJECT/
├── Data/
├── Data_prep/
├── mlruns/
├── app.py
├── requirements.txt
├── Project_documentation.pdf
└── README.md
```

---

## ⚠️ Important Notes

* Large datasets are not included due to GitHub size limits
* Ensure datasets are downloaded before running locally
* MLflow logs are included for experiment reference

---

## 🚀 Deployment

This application is deployed on **Streamlit Cloud** with GitHub integration for continuous updates.

👉 https://chicago-crime-analysis-jatjuq9fu8kdjydgr2qkmh.streamlit.app/

---

## 📌 Conclusion

This project demonstrates a **complete end-to-end data science workflow**:
from data preprocessing → feature engineering → clustering → experiment tracking → deployment.

It combines machine learning with interactive visualization to generate **actionable insights for public safety analysis**.

---

## 👤 Author

**Rama Sekar**

Aspiring Data Scientist | Python | Machine Learning | Streamlit

---
