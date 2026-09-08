import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
import statsmodels.api as sm  
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 1. Page Configuration
st.set_page_config(page_title="Medical Insurance Analysis", layout="wide")
st.title("Medical Insurance Costs: Data Science Dashboard")

# 2. Load Data (Cached so it doesn't reload on every interaction)
@st.cache_data
def load_data():
    # Make sure 'insurance.csv' is in the same folder, or update the path
    df = pd.read_csv("data/insurance.csv")
    return df

df = load_data()

# 3. Create the three tabs
tab1, tab2, tab3 = st.tabs([
    "Data Exploration", 
    "Hypothesis Testing Lab", 
    "Live Prediction & Diagnostics"
])

# ==========================================
# TAB 1: DATA EXPLORATION
# ==========================================
with tab1:
    st.header("Exploratory Data Analysis")
    
    # --- Sidebar Filters ---
    st.sidebar.header("Filter Data")
    age_range = st.sidebar.slider("Select Age Range", int(df['age'].min()), int(df['age'].max()), (18, 64))
    selected_smoker = st.sidebar.multiselect("Select Smoker Status", options=df['smoker'].unique(), default=df['smoker'].unique())
    selected_region = st.sidebar.multiselect("Select Region", options=df['region'].unique(), default=df['region'].unique())
    
    # Apply filters to dataframe
    filtered_df = df[
        (df['age'] >= age_range[0]) & (df['age'] <= age_range[1]) &
        (df['smoker'].isin(selected_smoker)) &
        (df['region'].isin(selected_region))
    ]
    
    st.write(f"Data shape after filtering: **{filtered_df.shape[0]} rows**")
    
    # --- Descriptive Metrics ---
    st.subheader("Descriptive Statistics")
    # Get numeric columns
    num_cols = filtered_df.select_dtypes(include=[np.number])
    
    # Calculate required metrics
    desc_stats = pd.DataFrame({
        'Mean': num_cols.mean(),
        'Median': num_cols.median(),
        'Std Dev': num_cols.std(),
        'IQR': num_cols.quantile(0.75) - num_cols.quantile(0.25),
        'Skewness': num_cols.skew(),
        'Kurtosis': num_cols.kurtosis()
    })
    st.dataframe(desc_stats.T) # Transposed for better readability
    
    # --- Visual Exploration ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Distribution of Charges")
        # Histogram using Plotly
        fig_hist = px.histogram(filtered_df, x="charges", color="smoker", marginal="box", 
                                title="Medical Charges Distribution", opacity=0.7)
        st.plotly_chart(fig_hist, use_container_width=True)
        
    with col2:
        st.subheader("Bivariate Scatter Plot")
        # Scatter plot using Plotly
        fig_scatter = px.scatter(filtered_df, x="bmi", y="charges", color="smoker", 
                                 title="BMI vs Medical Charges", trendline="ols")
        st.plotly_chart(fig_scatter, use_container_width=True)
        
    st.subheader("Correlation Matrix")
    # Correlation Matrix using Seaborn & Matplotlib
    fig_corr, ax = plt.subplots(figsize=(8, 4))
    sns.heatmap(num_cols.corr(), annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
    st.pyplot(fig_corr)

    # ==========================================
# TAB 2: HYPOTHESIS TESTING LAB
# ==========================================
with tab2:
    st.header("Hypothesis Testing Lab")
    
    # --- UI: User Selections ---
    col1, col2 = st.columns(2)
    with col1:
        # Choose a binary categorical variable for the 2-sample test
        cat_var = st.selectbox("Select Categorical Factor (2 Groups)", ["smoker", "sex"])
    with col2:
        # Choose a numerical metric to compare
        num_var = st.selectbox("Select Numerical Metric to Compare", ["charges", "bmi", "age"])
    
    st.divider()
    
    # --- TEST 1: Two-Sample Comparison ---
    st.subheader(f"Test 1: Does '{num_var}' differ significantly by '{cat_var}'?")
    
    # Split the data into two groups based on user selection
    groups = df[cat_var].unique()
    group_A = df[df[cat_var] == groups[0]][num_var]
    group_B = df[df[cat_var] == groups[1]][num_var]
    
    st.write(f"**Group A ({groups[0]}):** {len(group_A)} observations")
    st.write(f"**Group B ({groups[1]}):** {len(group_B)} observations")
    
    # 1. Check Normality (Shapiro-Wilk)
    stat_shapiro_A, p_shapiro_A = stats.shapiro(group_A)
    stat_shapiro_B, p_shapiro_B = stats.shapiro(group_B)
    is_normal = (p_shapiro_A > 0.05) and (p_shapiro_B > 0.05)
    
    # 2. Check Variance (Levene's Test)
    stat_levene, p_levene = stats.levene(group_A, group_B)
    equal_var = p_levene > 0.05
    
    # 3. Execute appropriate test based on assumptions
    if is_normal:
        st.info("Data appears normally distributed (Shapiro-Wilk p > 0.05). Running Two-Sample t-test.")
        test_stat, p_val = stats.ttest_ind(group_A, group_B, equal_var=equal_var)
        test_name = "Independent t-test"
    else:
        st.warning("Data violates normality assumption. Running non-parametric Mann-Whitney U test.")
        test_stat, p_val = stats.mannwhitneyu(group_A, group_B)
        test_name = "Mann-Whitney U test"
        
    # 4. Render Conclusion
    st.write(f"**Test Statistic:** {test_stat:.4f}")
    st.write(f"**p-value:** {p_val:.4e}")
    
    if p_val < 0.05:
        st.success(f"**Conclusion:** Reject H0 at α = 0.05. There is a significant difference in {num_var} between {groups[0]} and {groups[1]}.")
    else:
        st.error(f"**Conclusion:** Fail to Reject H0 at α = 0.05. No significant difference in {num_var} between {groups[0]} and {groups[1]} was found.")

    st.divider()

    # --- TEST 2: One-Way ANOVA (Comparing across Regions) ---
    st.subheader(f"Test 2: One-Way ANOVA - Does '{num_var}' differ across Regions?")
    st.write("Testing if the average metric differs across the 4 geographic regions (northeast, northwest, southeast, southwest).")
    
    regions = [df[df['region'] == r][num_var] for r in df['region'].unique()]
    f_stat, p_anova = stats.f_oneway(*regions)
    
    st.write(f"**F-Statistic:** {f_stat:.4f}")
    st.write(f"**p-value:** {p_anova:.4e}")
    
    if p_anova < 0.05:
        st.success(f"**Conclusion:** Reject H0 at α = 0.05. There is a significant difference in {num_var} across regions.")
    else:
        st.error(f"**Conclusion:** Fail to Reject H0 at α = 0.05. No significant difference in {num_var} across regions.")


# ==========================================
# TAB 3: LIVE PREDICTION & DIAGNOSTICS
# ==========================================
with tab3:
    st.header("Live Prediction & Model Diagnostics")
    
    # --- 1. Data Preprocessing & Model Formulation ---
    # Convert categorical text data into dummy variables (0s and 1s)
    df_encoded = pd.get_dummies(df, columns=['sex', 'smoker', 'region'], drop_first=True)
    
    # Define independent variables (X) and dependent variable (y)
    X = df_encoded.drop('charges', axis=1)
    X = sm.add_constant(X) # Adds the beta_0 intercept
    X = X.astype(float)  
    y = df_encoded['charges'].astype(float) 

    
    # Fit the OLS Model
    model = sm.OLS(y, X).fit()
    
    # --- 2. Live Prediction Interactive UI ---
    st.subheader("Interactive Cost Predictor")
    st.write("Adjust the patient details below to generate a real-time prediction.")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        input_age = st.slider("Age", int(df['age'].min()), int(df['age'].max()), 30)
        input_bmi = st.number_input("BMI (Body Mass Index)", float(df['bmi'].min()), float(df['bmi'].max()), 25.0)
    with col2:
        input_children = st.slider("Number of Children", 0, 5, 0)
        input_sex = st.selectbox("Sex", ["male", "female"])
    with col3:
        input_smoker = st.selectbox("Smoker?", ["yes", "no"])
        input_region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])
        
    # Construct a dataframe for the user's input
    user_data = pd.DataFrame({
        'age': [input_age], 'bmi': [input_bmi], 'children': [input_children],
        'sex': [input_sex], 'smoker': [input_smoker], 'region': [input_region]
    })
    
    # Encode the user input exactly like we encoded the training data
    user_encoded = pd.get_dummies(user_data)
    user_encoded = user_encoded.reindex(columns=X.columns, fill_value=0)
    user_encoded['const'] = 1.0 # Ensure intercept is set
    user_encoded = user_encoded.astype(float)
    
    # Generate Prediction and 95% Interval
    prediction = model.get_prediction(user_encoded)
    pred_summary = prediction.summary_frame(alpha=0.05)
    
    st.success(f"**Predicted Medical Charge:** ${pred_summary['mean'].values[0]:,.2f}")
    st.write(f"**95% Prediction Interval:** ${pred_summary['obs_ci_lower'].values[0]:,.2f} to ${pred_summary['obs_ci_upper'].values[0]:,.2f}")
    
    st.divider()
    
    # --- 3. Model Parameters & Diagnostics ---
    st.subheader("Model Parameters & Gauss-Markov Diagnostics")
    
    # Display R-squared values
    st.write(f"**Overall Model Fit:** R² = {model.rsquared:.4f} | Adjusted R² = {model.rsquared_adj:.4f}")
    
    # Display Coefficients and p-values
    st.write("**Estimated Coefficients (Beta parameters):**")
    params_df = pd.DataFrame({
        "Coefficient": model.params,
        "p-value": model.pvalues,
        "95% CI Lower": model.conf_int()[0],
        "95% CI Upper": model.conf_int()[1]
    })
    st.dataframe(params_df)
    
    # Diagnostic Plots (Linearity & Normality)
    col_diag1, col_diag2 = st.columns(2)
    
    with col_diag1:
        st.write("**1. Linearity & Homoscedasticity**")
        st.caption("Residuals vs. Fitted Values")
        fig_resid, ax1 = plt.subplots(figsize=(6, 4))
        sns.scatterplot(x=model.fittedvalues, y=model.resid, ax=ax1, alpha=0.6, color="blue")
        ax1.axhline(0, color='red', linestyle='--')
        ax1.set_xlabel("Fitted Values")
        ax1.set_ylabel("Residuals")
        st.pyplot(fig_resid)
        
    with col_diag2:
        st.write("**2. Normality of Residuals**")
        st.caption("Q-Q Plot")
        fig_qq, ax2 = plt.subplots(figsize=(6, 4))
        sm.qqplot(model.resid, line='45', fit=True, ax=ax2)
        st.pyplot(fig_qq)
        
    # Multicollinearity (VIF) for continuous predictors
    st.write("**3. Multicollinearity Checks**")
    st.caption("Variance Inflation Factor (VIF) for continuous predictors. Values > 5 indicate high multicollinearity.")
    
    continuous_features = df[['age', 'bmi', 'children']]
    vif_data = pd.DataFrame()
    vif_data["Feature"] = continuous_features.columns
    # Calculate VIF for each continuous feature
    vif_data["VIF"] = [variance_inflation_factor(continuous_features.values, i) for i in range(len(continuous_features.columns))]
    
    st.dataframe(vif_data)