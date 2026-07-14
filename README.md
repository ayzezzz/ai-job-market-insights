# AI Job Market Insights & Salary Predictor

An end-to-end data science project that analyzes the AI and Data Science job market and predicts salaries using a machine learning model.

The project combines exploratory data analysis (EDA), a Random Forest regression model, and an interactive Streamlit dashboard for real-time salary prediction.

---

## 📌 Features

- Analyze salary trends across job titles, industries, and experience levels.
- Explore the highest-paying AI roles and the most in-demand skills.
- Predict annual salaries using a trained Random Forest model.
- Interactive Streamlit dashboard for data exploration and prediction.

---

## 🛠️ Tech Stack

- Python
- Pandas & NumPy
- Matplotlib & Seaborn
- Scikit-learn
- Streamlit
- Git & GitHub

---

## 📂 Project Structure

```text
├── analysis.ipynb
├── app.py
├── ai_job_dataset.csv
├── salary_predictor_model.pkl
├── model_columns.pkl
├── requirements.txt
└── README.md
```

---

## ⚙️ Run Locally

```bash
git clone https://github.com/ayzezzz/ai-job-market-insights.git
cd ai-job-market-insights

python -m venv .venv

# macOS / Linux
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

---
