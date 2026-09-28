import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from scipy import stats

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
    classification_report
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Employee Attrition Analytics",
    page_icon="📊",
    layout="wide"
)


# ============================================================
# LOAD DATASET
# ============================================================

DATA_PATH = "dataset/employee_attrition.csv"

if not os.path.exists(DATA_PATH):
    st.error("Dataset not found: dataset/employee_attrition.csv")
    st.stop()

df = pd.read_csv(DATA_PATH)


# ============================================================
# TITLE
# ============================================================

st.title("📊 AI-Driven Employee Attrition Prediction")
st.subheader(
    "Statistical Machine Learning and Workforce Analytics"
)

st.caption(
    "BAD702 – Statistical Machine Learning for Data Science"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("📌 Project Navigation")

page = st.sidebar.radio(
    "Select Section",
    [
        "Project Overview",
        "Dataset Analysis",
        "Exploratory Data Analysis",
        "Statistical Analysis",
        "Regression Analysis",
        "Machine Learning",
        "Model Comparison",
        "Attrition Prediction"
    ]
)


# ============================================================
# PROJECT OVERVIEW
# ============================================================

if page == "Project Overview":

    st.header("📌 Project Overview")

    st.write("""
    This project analyzes employee attrition using statistical analysis
    and machine learning techniques. The objective is to identify factors
    associated with employee attrition and develop classification models
    for predicting whether an employee may leave the organization.
    """)

    st.subheader("🎯 Objectives")

    objectives = [
        "Clean and preprocess the employee dataset.",
        "Perform exploratory data analysis.",
        "Study sampling distributions and confidence intervals.",
        "Perform hypothesis testing and A/B testing.",
        "Build simple and multiple regression models.",
        "Apply Linear Discriminant Analysis.",
        "Build Logistic Regression and Random Forest models.",
        "Compare classification model performance.",
        "Develop a live employee attrition prediction system."
    ]

    for objective in objectives:
        st.write("• " + objective)

    st.subheader("🔄 Project Workflow")

    st.code("""
Raw Dataset
     ↓
Data Cleaning
     ↓
Exploratory Data Analysis
     ↓
Statistical Analysis
     ↓
Hypothesis Testing
     ↓
Regression Analysis
     ↓
Discriminant Analysis
     ↓
Machine Learning
     ↓
Model Evaluation
     ↓
Employee Attrition Prediction
""")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Rows", df.shape[0])

    with col2:
        st.metric("Columns", df.shape[1])

    with col3:
        st.metric(
            "Attrition Cases",
            (df["Attrition"] == "Yes").sum()
        )

    with col4:
        st.metric(
            "Attrition Rate",
            f"{(df['Attrition'] == 'Yes').mean()*100:.2f}%"
        )


# ============================================================
# DATASET ANALYSIS
# ============================================================

elif page == "Dataset Analysis":

    st.header("📂 Dataset Analysis")

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Records", df.shape[0])

    with col2:
        st.metric("Features", df.shape[1])

    with col3:
        st.metric(
            "Duplicate Rows",
            df.duplicated().sum()
        )

    st.subheader("Dataset Information")

    info_df = pd.DataFrame({
        "Column": df.columns,
        "Data Type": df.dtypes.astype(str),
        "Missing Values": df.isnull().sum().values,
        "Unique Values": df.nunique().values
    })

    st.dataframe(
        info_df,
        use_container_width=True
    )

    st.subheader("Descriptive Statistics")

    st.dataframe(
        df.describe(),
        use_container_width=True
    )


# ============================================================
# EDA
# ============================================================

elif page == "Exploratory Data Analysis":

    st.header("📈 Exploratory Data Analysis")

    st.subheader("Attrition Distribution")

    attrition_counts = df["Attrition"].value_counts()

    fig, ax = plt.subplots()

    attrition_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Attrition")
    ax.set_ylabel("Number of Employees")
    ax.set_title("Employee Attrition Distribution")

    st.pyplot(fig)

    st.subheader("Age Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df["Age"],
        bins=20
    )

    ax.set_xlabel("Age")
    ax.set_ylabel("Frequency")
    ax.set_title("Age Distribution")

    st.pyplot(fig)

    st.subheader("Age Boxplot")

    fig, ax = plt.subplots()

    ax.boxplot(df["Age"])

    ax.set_ylabel("Age")
    ax.set_title("Age Boxplot")

    st.pyplot(fig)

    st.subheader("Department Distribution")

    department_counts = df["Department"].value_counts()

    fig, ax = plt.subplots()

    department_counts.plot(
        kind="bar",
        ax=ax
    )

    ax.set_xlabel("Department")
    ax.set_ylabel("Employees")

    st.pyplot(fig)

    st.subheader("Monthly Income Distribution")

    fig, ax = plt.subplots()

    ax.hist(
        df["MonthlyIncome"],
        bins=30
    )

    ax.set_xlabel("Monthly Income")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    st.subheader("Age vs Monthly Income")

    fig, ax = plt.subplots()

    for value in ["No", "Yes"]:

        subset = df[df["Attrition"] == value]

        ax.scatter(
            subset["Age"],
            subset["MonthlyIncome"],
            label=value,
            alpha=0.6
        )

    ax.set_xlabel("Age")
    ax.set_ylabel("Monthly Income")
    ax.legend()

    st.pyplot(fig)

    st.subheader("Correlation Matrix")

    numeric_df = df.select_dtypes(
        include=["int64", "float64"]
    )

    corr = numeric_df.corr()

    fig, ax = plt.subplots(
        figsize=(12, 8)
    )

    im = ax.imshow(
        corr,
        aspect="auto"
    )

    ax.set_xticks(
        range(len(corr.columns))
    )

    ax.set_yticks(
        range(len(corr.columns))
    )

    ax.set_xticklabels(
        corr.columns,
        rotation=90
    )

    ax.set_yticklabels(
        corr.columns
    )

    fig.colorbar(im)

    st.pyplot(fig)


# ============================================================
# STATISTICAL ANALYSIS
# ============================================================

elif page == "Statistical Analysis":

    st.header("📊 Statistical Analysis")

    population = df["MonthlyIncome"].dropna()

    st.subheader("Population Statistics")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Population Size", len(population))

    with col2:
        st.metric(
            "Mean Monthly Income",
            f"{population.mean():.2f}"
        )

    with col3:
        st.metric(
            "Standard Deviation",
            f"{population.std():.2f}"
        )

    st.subheader("Sampling Distribution")

    sample = population.sample(
        n=100,
        random_state=42
    )

    sample_means = []

    for i in range(1000):

        repeated_sample = population.sample(
            n=100,
            random_state=i
        )

        sample_means.append(
            repeated_sample.mean()
        )

    fig, ax = plt.subplots()

    ax.hist(
        sample_means,
        bins=30
    )

    ax.set_xlabel("Sample Mean")
    ax.set_ylabel("Frequency")

    st.pyplot(fig)

    st.subheader("95% Confidence Interval")

    sample_mean = sample.mean()
    sample_std = sample.std()
    sample_size = len(sample)

    standard_error = (
        sample_std /
        np.sqrt(sample_size)
    )

    t_critical = stats.t.ppf(
        0.975,
        sample_size - 1
    )

    margin_error = (
        t_critical *
        standard_error
    )

    lower = sample_mean - margin_error
    upper = sample_mean + margin_error

    st.write(
        f"Sample Mean: **{sample_mean:.2f}**"
    )

    st.write(
        f"95% Confidence Interval: "
        f"**({lower:.2f}, {upper:.2f})**"
    )

    st.subheader("Hypothesis Testing")

    income_left = df[
        df["Attrition"] == "Yes"
    ]["MonthlyIncome"]

    income_stayed = df[
        df["Attrition"] == "No"
    ]["MonthlyIncome"]

    t_stat, p_value = stats.ttest_ind(
        income_left,
        income_stayed,
        equal_var=False
    )

    st.write(
        f"t-statistic: **{t_stat:.4f}**"
    )

    st.write(
        f"p-value: **{p_value:.6f}**"
    )

    st.write(
        "H₀: There is no significant difference in mean "
        "Monthly Income between employees who stayed and left."
    )

    st.write(
        "H₁: There is a significant difference in mean "
        "Monthly Income between employees who stayed and left."
    )


# ============================================================
# REGRESSION ANALYSIS
# ============================================================

elif page == "Regression Analysis":

    st.header("📉 Regression Analysis")

    st.subheader(
        "Simple Linear Regression: Age → Monthly Income"
    )

    X = df[["Age"]]
    y = df["MonthlyIncome"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    simple_model = LinearRegression()

    simple_model.fit(
        X_train,
        y_train
    )

    prediction = simple_model.predict(
        X_test
    )

    r2 = r2_score(
        y_test,
        prediction
    )

    mae = mean_absolute_error(
        y_test,
        prediction
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            prediction
        )
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("R²", f"{r2:.4f}")

    with col2:
        st.metric("MAE", f"{mae:.2f}")

    with col3:
        st.metric("RMSE", f"{rmse:.2f}")

    st.subheader("Multiple Linear Regression")

    features = [
        "Age",
        "JobLevel",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsWithCurrManager"
    ]

    X_multi = df[features]

    y_multi = df["MonthlyIncome"]

    X_train_m, X_test_m, y_train_m, y_test_m = train_test_split(
        X_multi,
        y_multi,
        test_size=0.20,
        random_state=42
    )

    multiple_model = LinearRegression()

    multiple_model.fit(
        X_train_m,
        y_train_m
    )

    prediction_m = multiple_model.predict(
        X_test_m
    )

    st.write(
        f"R² Score: **{r2_score(y_test_m, prediction_m):.4f}**"
    )

    st.write(
        f"MAE: **{mean_absolute_error(y_test_m, prediction_m):.2f}**"
    )

    st.write(
        f"RMSE: **{np.sqrt(mean_squared_error(y_test_m, prediction_m)):.2f}**"
    )


# ============================================================
# MACHINE LEARNING
# ============================================================

elif page == "Machine Learning":

    st.header("🤖 Machine Learning")

    y_ml = df["Attrition"].map({
        "No": 0,
        "Yes": 1
    })

    X_ml = df.drop(
        columns=[
            "Attrition",
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours"
        ],
        errors="ignore"
    )

    numerical = X_ml.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical = X_ml.select_dtypes(
        include=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]),
                numerical
            ),
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]),
                categorical
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_ml,
        y_ml,
        test_size=0.20,
        random_state=42,
        stratify=y_ml
    )

    logistic_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    logistic_model.fit(
        X_train,
        y_train
    )

    logistic_pred = logistic_model.predict(
        X_test
    )

    random_forest_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ])

    random_forest_model.fit(
        X_train,
        y_train
    )

    rf_pred = random_forest_model.predict(
        X_test
    )

    st.subheader("Logistic Regression")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy_score(y_test, logistic_pred):.3f}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision_score(y_test, logistic_pred, zero_division=0):.3f}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall_score(y_test, logistic_pred, zero_division=0):.3f}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1_score(y_test, logistic_pred, zero_division=0):.3f}"
        )

    st.subheader("Random Forest")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            f"{accuracy_score(y_test, rf_pred):.3f}"
        )

    with col2:
        st.metric(
            "Precision",
            f"{precision_score(y_test, rf_pred, zero_division=0):.3f}"
        )

    with col3:
        st.metric(
            "Recall",
            f"{recall_score(y_test, rf_pred, zero_division=0):.3f}"
        )

    with col4:
        st.metric(
            "F1 Score",
            f"{f1_score(y_test, rf_pred, zero_division=0):.3f}"
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

elif page == "Model Comparison":

    st.header("📊 Model Comparison")

    y = df["Attrition"].map({
        "No": 0,
        "Yes": 1
    })

    X = df.drop(
        columns=[
            "Attrition",
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours"
        ],
        errors="ignore"
    )

    numerical = X.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]),
                numerical
            ),
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]),
                categorical
            )
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    lr_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                random_state=42
            )
        )
    ])

    lr_model.fit(
        X_train,
        y_train
    )

    lr_pred = lr_model.predict(
        X_test
    )

    rf_model = Pipeline([
        (
            "preprocessor",
            preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ])

    rf_model.fit(
        X_train,
        y_train
    )

    rf_pred = rf_model.predict(
        X_test
    )

    lda_features = [
        "Age",
        "MonthlyIncome",
        "JobLevel",
        "JobSatisfaction",
        "JobInvolvement",
        "PerformanceRating",
        "TotalWorkingYears",
        "YearsAtCompany",
        "YearsInCurrentRole",
        "YearsWithCurrManager"
    ]

    X_lda = df[lda_features]

    X_lda_train, X_lda_test, y_lda_train, y_lda_test = train_test_split(
        X_lda,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    lda_model = LinearDiscriminantAnalysis()

    lda_model.fit(
        X_lda_train,
        y_lda_train
    )

    lda_pred = lda_model.predict(
        X_lda_test
    )

    comparison = pd.DataFrame({
        "Model": [
            "LDA",
            "Logistic Regression",
            "Random Forest"
        ],
        "Accuracy": [
            accuracy_score(
                y_lda_test,
                lda_pred
            ),
            accuracy_score(
                y_test,
                lr_pred
            ),
            accuracy_score(
                y_test,
                rf_pred
            )
        ],
        "Precision": [
            precision_score(
                y_lda_test,
                lda_pred,
                zero_division=0
            ),
            precision_score(
                y_test,
                lr_pred,
                zero_division=0
            ),
            precision_score(
                y_test,
                rf_pred,
                zero_division=0
            )
        ],
        "Recall": [
            recall_score(
                y_lda_test,
                lda_pred,
                zero_division=0
            ),
            recall_score(
                y_test,
                lr_pred,
                zero_division=0
            ),
            recall_score(
                y_test,
                rf_pred,
                zero_division=0
            )
        ],
        "F1-Score": [
            f1_score(
                y_lda_test,
                lda_pred,
                zero_division=0
            ),
            f1_score(
                y_test,
                lr_pred,
                zero_division=0
            ),
            f1_score(
                y_test,
                rf_pred,
                zero_division=0
            )
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    comparison.set_index(
        "Model"
    )[[
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score"
    ]].plot(
        kind="bar",
        ax=ax
    )

    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title(
        "Classification Model Comparison"
    )

    st.pyplot(fig)


# ============================================================
# ATTRITION PREDICTION
# ============================================================

elif page == "Attrition Prediction":

    st.header("🔮 Employee Attrition Prediction")

    st.write(
        "Enter employee details below and use the trained "
        "Random Forest model to predict employee attrition."
    )

    st.info(
        "The prediction uses the complete dataset feature set, "
        "including Hourly Rate and Monthly Rate."
    )

    # --------------------------------------------------------
    # TRAIN RANDOM FOREST
    # --------------------------------------------------------

    prediction_df = pd.read_csv(
        "dataset/employee_attrition.csv"
    )

    y_prediction = prediction_df[
        "Attrition"
    ].map({
        "No": 0,
        "Yes": 1
    })

    X_prediction = prediction_df.drop(
        columns=[
            "Attrition",
            "EmployeeCount",
            "EmployeeNumber",
            "Over18",
            "StandardHours"
        ],
        errors="ignore"
    )

    numerical_prediction = X_prediction.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    categorical_prediction = X_prediction.select_dtypes(
        include=["object"]
    ).columns.tolist()

    prediction_preprocessor = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="median"
                        )
                    ),
                    (
                        "scaler",
                        StandardScaler()
                    )
                ]),
                numerical_prediction
            ),
            (
                "cat",
                Pipeline([
                    (
                        "imputer",
                        SimpleImputer(
                            strategy="most_frequent"
                        )
                    ),
                    (
                        "encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]),
                categorical_prediction
            )
        ]
    )

    prediction_model = Pipeline([
        (
            "preprocessor",
            prediction_preprocessor
        ),
        (
            "classifier",
            RandomForestClassifier(
                n_estimators=200,
                random_state=42
            )
        )
    ])

    prediction_model.fit(
        X_prediction,
        y_prediction
    )

    # --------------------------------------------------------
    # EMPLOYEE INPUT
    # --------------------------------------------------------

    st.subheader("👤 Employee Details")

    with st.form("prediction_form"):

        col1, col2 = st.columns(2)

        # ----------------------------------------------------
        # COLUMN 1
        # ----------------------------------------------------

        with col1:

            age = st.number_input(
                "Age",
                min_value=18,
                max_value=70,
                value=30
            )

            business_travel = st.selectbox(
                "Business Travel",
                prediction_df["BusinessTravel"].unique()
            )

            daily_rate = st.number_input(
                "Daily Rate",
                min_value=100,
                max_value=1500,
                value=800
            )

            department = st.selectbox(
                "Department",
                prediction_df["Department"].unique()
            )

            distance_from_home = st.number_input(
                "Distance From Home",
                min_value=1,
                max_value=100,
                value=5
            )

            education = st.slider(
                "Education",
                1,
                5,
                3
            )

            education_field = st.selectbox(
                "Education Field",
                prediction_df["EducationField"].unique()
            )

            environment_satisfaction = st.slider(
                "Environment Satisfaction",
                1,
                4,
                3
            )

            gender = st.selectbox(
                "Gender",
                prediction_df["Gender"].unique()
            )

            job_involvement = st.slider(
                "Job Involvement",
                1,
                4,
                3
            )

            job_level = st.slider(
                "Job Level",
                1,
                5,
                2
            )

            job_role = st.selectbox(
                "Job Role",
                prediction_df["JobRole"].unique()
            )

            job_satisfaction = st.slider(
                "Job Satisfaction",
                1,
                4,
                3
            )

            marital_status = st.selectbox(
                "Marital Status",
                prediction_df["MaritalStatus"].unique()
            )

        # ----------------------------------------------------
        # COLUMN 2
        # ----------------------------------------------------

        with col2:

            monthly_income = st.number_input(
                "Monthly Income",
                min_value=1000,
                max_value=50000,
                value=5000
            )

            # FIXED: HOURLY RATE
            hourly_rate = st.number_input(
                "Hourly Rate",
                min_value=1,
                max_value=200,
                value=65
            )

            # FIXED: MONTHLY RATE
            monthly_rate = st.number_input(
                "Monthly Rate",
                min_value=1000,
                max_value=30000,
                value=15000
            )

            num_companies = st.number_input(
                "Number of Companies Worked",
                min_value=0,
                max_value=20,
                value=2
            )

            overtime = st.selectbox(
                "OverTime",
                prediction_df["OverTime"].unique()
            )

            percent_salary_hike = st.slider(
                "Percent Salary Hike",
                10,
                30,
                15
            )

            performance_rating = st.slider(
                "Performance Rating",
                1,
                4,
                3
            )

            relationship_satisfaction = st.slider(
                "Relationship Satisfaction",
                1,
                4,
                3
            )

            stock_option_level = st.slider(
                "Stock Option Level",
                0,
                3,
                1
            )

            total_working_years = st.number_input(
                "Total Working Years",
                min_value=0,
                max_value=50,
                value=8
            )

            training_times_last_year = st.number_input(
                "Training Times Last Year",
                min_value=0,
                max_value=10,
                value=3
            )

            work_life_balance = st.slider(
                "Work Life Balance",
                1,
                4,
                3
            )

            years_at_company = st.number_input(
                "Years At Company",
                min_value=0,
                max_value=50,
                value=5
            )

            years_in_current_role = st.number_input(
                "Years In Current Role",
                min_value=0,
                max_value=20,
                value=3
            )

            years_since_last_promotion = st.number_input(
                "Years Since Last Promotion",
                min_value=0,
                max_value=20,
                value=1
            )

            years_with_curr_manager = st.number_input(
                "Years With Current Manager",
                min_value=0,
                max_value=20,
                value=3
            )

        submitted = st.form_submit_button(
            "🔍 Predict Employee Attrition"
        )

    # --------------------------------------------------------
    # REAL PREDICTION
    # --------------------------------------------------------

    if submitted:

        employee = pd.DataFrame([{

            "Age": age,

            "BusinessTravel": business_travel,

            "DailyRate": daily_rate,

            "Department": department,

            "DistanceFromHome": distance_from_home,

            "Education": education,

            "EducationField": education_field,

            "EnvironmentSatisfaction":
                environment_satisfaction,

            "Gender": gender,

            "JobInvolvement":
                job_involvement,

            "JobLevel": job_level,

            "JobRole": job_role,

            "JobSatisfaction":
                job_satisfaction,

            "MaritalStatus":
                marital_status,

            "MonthlyIncome":
                monthly_income,

            # FIXED COLUMNS
            "HourlyRate":
                hourly_rate,

            "MonthlyRate":
                monthly_rate,

            "NumCompaniesWorked":
                num_companies,

            "OverTime":
                overtime,

            "PercentSalaryHike":
                percent_salary_hike,

            "PerformanceRating":
                performance_rating,

            "RelationshipSatisfaction":
                relationship_satisfaction,

            "StockOptionLevel":
                stock_option_level,

            "TotalWorkingYears":
                total_working_years,

            "TrainingTimesLastYear":
                training_times_last_year,

            "WorkLifeBalance":
                work_life_balance,

            "YearsAtCompany":
                years_at_company,

            "YearsInCurrentRole":
                years_in_current_role,

            "YearsSinceLastPromotion":
                years_since_last_promotion,

            "YearsWithCurrManager":
                years_with_curr_manager
        }])

        # ----------------------------------------------------
        # SAFETY CHECK
        # ----------------------------------------------------

        missing_columns = set(
            X_prediction.columns
        ) - set(
            employee.columns
        )

        if missing_columns:

            st.error(
                f"Missing columns: {missing_columns}"
            )

        else:

            # Reorder columns exactly like training data
            employee = employee[
                X_prediction.columns
            ]

            # Prediction
            prediction = prediction_model.predict(
                employee
            )[0]

            probability = prediction_model.predict_proba(
                employee
            )[0][1]

            st.divider()

            st.subheader("🎯 Prediction Result")

            if prediction == 1:

                st.error(
                    "⚠️ Predicted Attrition: YES"
                )

            else:

                st.success(
                    "✅ Predicted Attrition: NO"
                )

            st.metric(
                "Estimated Attrition Probability",
                f"{probability * 100:.2f}%"
            )

            st.progress(
                float(probability)
            )

            st.write(
                "The prediction is generated using the "
                "**Random Forest classification model**."
            )

            st.caption(
                "This is a machine-learning estimate based "
                "on patterns in the dataset and is not a certainty."
            )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.info(
    "BAD702 Statistical Machine Learning Project"
)

st.sidebar.write(
    "Employee Attrition Prediction & Workforce Analytics"
)