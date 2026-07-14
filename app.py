import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

st.set_page_config(
    page_title="AI Job Market & Salary Predictor",
    layout="wide"
)

@st.cache_data
def load_data():
    return pd.read_csv("ai_job_dataset.csv")

try:
    df = load_data()
    model = joblib.load("salary_predictor_model.pkl")
    model_columns = joblib.load("model_columns.pkl")
except Exception as e:
    st.error(f"Error loading files: {e}")
    st.stop()

st.title("AI & Data Science Job Market Portal")
st.write(
    "Explore global job market trends and predict industry-standard salaries using Machine Learning."
)

tab1, tab2 = st.tabs(["📊 Market Analytics", "🔮 Salary Predictor"])


with tab1:

    st.header("Global AI & Data Science Job Market Insights")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric(
        "📄 Total Job Listings",
        f"{len(df):,}"
    )

    col2.metric(
        "💰 Average Salary",
        f"${df['salary_usd'].mean():,.0f}"
    )

    col3.metric(
        "🌍 Countries",
        df["company_location"].nunique()
    )

    col4.metric(
    "👨‍💻 Job Roles",
    df["job_title"].nunique()
)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Top 10 Highest Paying Roles")

        top_paying = (
            df.groupby("job_title")["salary_usd"]
            .mean()
            .sort_values(ascending=False)
            .head(10)
        )

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(
            x=top_paying.values,
            y=top_paying.index,
            palette="viridis",
            ax=ax,
        )
        ax.set_xlabel("Average Salary (USD)")
        ax.set_ylabel("Job Title")

        st.pyplot(fig)

    with col2:

        st.subheader("Salary Distribution by Experience Level")

        fig2, ax2 = plt.subplots(figsize=(10, 5))

        sns.boxplot(
            x="experience_level",
            y="salary_usd",
            data=df,
            order=["EN", "MI", "SE", "EX"],
            palette="coolwarm",
            ax=ax2,
        )

        ax2.set_xlabel("Experience Level")
        ax2.set_ylabel("Salary (USD)")

        st.pyplot(fig2)

    st.markdown("---")

    col3, col4 = st.columns(2)

    with col3:

        st.subheader("Top 10 Most In-Demand Skills")

        skills_series = df["required_skills"].dropna().str.split(",")
        exploded_skills = skills_series.explode().str.strip()

        top_skills = exploded_skills.value_counts().head(10)

        fig3, ax3 = plt.subplots(figsize=(10, 5))

        sns.barplot(
            x=top_skills.values,
            y=top_skills.index,
            palette="mako",
            ax=ax3,
        )

        ax3.set_xlabel("Demand Count")
        ax3.set_ylabel("Required Skill")

        st.pyplot(fig3)

    with col4:

        st.subheader("Experience vs Salary Correlation")

        fig4, ax4 = plt.subplots(figsize=(10, 5))

        sns.regplot(
            x="years_experience",
            y="salary_usd",
            data=df,
            scatter_kws={"alpha": 0.3},
            line_kws={"color": "red"},
            ax=ax4,
        )

        ax4.set_xlabel("Years of Experience")
        ax4.set_ylabel("Salary (USD)")

        st.pyplot(fig4)


with tab2:

    st.header("Predict Your Potential Salary")
    st.write(
        "Fill in the details below to estimate the market rate for this role."
    )

    col_input1, col_input2 = st.columns(2)

    experience_options = {
        "Entry Level": "EN",
        "Mid Level": "MI",
        "Senior": "SE",
        "Executive": "EX",
    }

    company_size_options = {
        "Small": "S",
        "Medium": "M",
        "Large": "L",
    }

    with col_input1:

        years_exp = st.slider(
            "Years of Experience",
            min_value=0,
            max_value=20,
            value=5,
        )

        selected_level = st.selectbox(
            "Experience Level",
            list(experience_options.keys()),
        )

        exp_level = experience_options[selected_level]

        selected_size = st.selectbox(
            "Company Size",
            list(company_size_options.keys()),
        )

        comp_size = company_size_options[selected_size]

    with col_input2:

        industry = st.selectbox(
            "Industry",
            sorted(df["industry"].dropna().unique()),
        )

        job_title = st.selectbox(
            "Job Title",
            sorted(df["job_title"].dropna().unique()),
        )

        company_location = st.selectbox(
            "Company Location",
            sorted(df["company_location"].dropna().unique()),
        )

        remote_ratio = st.selectbox(
            "Remote Ratio (%)",
            sorted(df["remote_ratio"].dropna().unique()),
        )

        employment_type = st.selectbox(
            "Employment Type",
            sorted(df["employment_type"].dropna().unique()),
        )

    if st.button("Calculate Estimated Salary", use_container_width=True):

        input_data = pd.DataFrame(
            [
                {
                    "years_experience": years_exp,
                    "experience_level": exp_level,
                    "company_size": comp_size,
                    "industry": industry,
                    "job_title": job_title,
                    "company_location": company_location,
                    "remote_ratio": remote_ratio,
                    "employment_type": employment_type,
                }
            ]
        )

        input_encoded = pd.get_dummies(input_data)
        input_encoded = input_encoded.reindex(
            columns=model_columns,
            fill_value=0,
        )

        prediction = model.predict(input_encoded)[0]

        st.success("Prediction completed successfully!")

        st.metric(
            label="💰 Estimated Annual Salary",
            value=f"${prediction:,.0f} USD",
        )