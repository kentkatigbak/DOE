# Import libraries
import streamlit as st
import pandas as pd
import itertools
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.stats.anova import anova_lm
import scipy.optimize as opt

# Page configurations
st.set_page_config(page_title="DOE | kentjkdigitals", layout="wide")
hide_st_style = """
                <style>
                #MainMenu {visibility:hidden;}
                footer {visibility:hidden;}
                header {visibility:hidden;}
                </style>
                """
st.markdown(hide_st_style, unsafe_allow_html=True)

# Remove top white space
st.markdown("""
        <style>
            .block-container {
                    padding-top: 0rem;
                    padding-bottom: 0rem;
                    padding-left: 1rem;
                    padding-right: 1rem;
                }
        </style>
        """, unsafe_allow_html=True)

# Page title and description

# Sidebar Menu
with st.sidebar:
    st.title("Design of Experiments")
    st.write("________________________")
    tool_selection = st.selectbox("Select DOE step:", ["App Introduction", "Experimental Design", "Graphical Analysis", "Response Optimizer"])
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    st.header("")
    st.write("Follow @kentjk on tiktok for more.")

if tool_selection == "App Introduction":
    st.header("App Introduction")
    st.subheader("Welcome to the Design of Experiments (DOE) Web App!")
    st.write("""This app is a powerful and user-friendly tool designed to streamline the planning,
            execution, and analysis of experiments. Whether you're optimizing a process, testing 
            multiple factors, or exploring relationships between variables, this app provides everything 
            you need to make data-driven decisions.

With features like automated design generation, statistical analysis, factorial plots, 
            and response optimization, our app empowers you to uncover significant insights quickly 
            and efficiently. Perfect for engineers, researchers, and quality professionals, the DOE 
            Web App ensures that your experiments are not only systematic but also impactful.

Start exploring the power of design of experiments today!""")
    
    st.markdown("#### ● Follow the following steps. Use sidebar dropdown menu to switch between steps.")
    st.subheader("1. Experimental Design")
    st.write(" - Define factors and levels and generate CSV file of experimental design.")
    st.subheader("2. Graphical Analysis")
    st.write(" - Generate graphical analysis of DOE results.")
    st.subheader("3. Response Optimizer")
    st.write(" - Generate regression formula from DOE results and predict response based on given levels of factors.")

# Experimental Design
if tool_selection == "Experimental Design":
    
    st.header("Experimental Design")
    # Step 1: Input Factors and Levels
    st.header("1. Define Factors and Levels")

    # Number of factors
    num_factors = st.number_input("Enter the number of factors:", min_value=1, step=1)

    # Input factors and levels
    factors = {}
    for i in range(num_factors):
        factor_name = st.text_input(f"Name of Factor {i+1}", key=f"factor_name_{i}")
        num_levels = st.number_input(f"Number of levels for {factor_name}", min_value=2, step=1, key=f"num_levels_{i}")
        
        # Level names
        levels = []
        for j in range(num_levels):
            level = st.text_input(f"Level {j+1} for {factor_name}", key=f"level_{i}_{j}")
            levels.append(level)
        factors[factor_name] = levels

    # Number of replicates
    num_replicates = st.number_input("Enter the number of replicates:", min_value=1, step=1, value=1)

    # Generate design button
    if st.button("Generate Design"):
        # Step 2: Create the Experimental Design
        st.header("2. Experimental Design")

        # Get all combinations of factor levels
        base_design = list(itertools.product(*factors.values()))

        # Apply replicates
        design = base_design * num_replicates

        # Create a DataFrame with an empty Response column and set the index to start at 1
        columns = list(factors.keys()) + ["Response"]
        df_design = pd.DataFrame(design, columns=factors.keys())
        df_design["Replicate"] = [i // len(base_design) + 1 for i in range(len(design))]  # Add replicate numbers
        df_design["Response"] = None  # Initialize response column with None
        
        # Set the index to start at 1 and rename it to "Run"
        df_design.index = range(1, len(df_design) + 1)
        df_design.index.name = "Run"

        # Display the design matrix
        st.write("Generated Design Matrix with Replicates:")
        st.dataframe(df_design)

        # Option to download the design matrix as CSV
        csv = df_design.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download Design Matrix with Replicates as CSV",
            data=csv,
            file_name='design_matrix_with_replicates.csv',
            mime='text/csv'
        )

# Graphical Analysis
if tool_selection == "Graphical Analysis":

    st.header("Graphical Analysis")
    # Step 1: Upload CSV file
    st.header("1. Upload CSV File")
    uploaded_file = st.file_uploader("Upload your DOE CSV file", type="csv")

    if uploaded_file:
        # Load the data
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", df)
        
        # Step 2: Display Factorial Analysis
        st.header("2. Factorial Analysis")
        
        # Prompt user to select response variable and factors
        response = st.selectbox("Select the Response Variable:", options=df.columns)
        factors = st.multiselect("Select Factors:", options=[col for col in df.columns if col != response])
        
        if factors:
            # Create factorial model formula
            formula = f"{response} ~ " + " + ".join(factors) + " + " + " + ".join([f"{i}:{j}" for i in factors for j in factors if i != j])
            
            # Fit the OLS model
            model = smf.ols(formula, data=df).fit()
            st.write(model.summary())

            # Dynamic interpretation of OLS results
            st.header("Interpretation of OLS Results")

            # Extract p-values and coefficients
            p_values = model.pvalues[1:]  # Exclude intercept
            coefficients = model.params[1:]  # Exclude intercept
            
            interpretations = []
            
            for factor, p_value, coefficient in zip(coefficients.index, p_values, coefficients):
                if p_value < 0.05:
                    significance = "significant"
                    effect_direction = "increase" if coefficient > 0 else "decrease"
                    interpretations.append(f"The factor '{factor}' is {significance} with a coefficient of {coefficient:.4f}, indicating that as this factor increases, the response variable tends to {effect_direction}.")
                else:
                    interpretations.append(f"The factor '{factor}' is not significant (p-value: {p_value:.4f}), suggesting it does not have a meaningful effect on the response variable.")
            
            # Display OLS interpretations
            for interpretation in interpretations:
                st.write(interpretation)
                
            # Display main effects and interaction plots
            st.header("Factorial Plots")
            
            # Main Effects Plot
            st.subheader("Main Effects Plot")
            fig, axes = plt.subplots(1, len(factors), figsize=(5 * len(factors), 4))
            for i, factor in enumerate(factors):
                sns.pointplot(data=df, x=factor, y=response, ax=axes[i] if len(factors) > 1 else axes)
                axes[i].set_title(f"Main Effect of {factor}")
            st.pyplot(fig)

            # Interaction Plot
            st.subheader("Interaction Plots")
            for i in range(len(factors)):
                for j in range(i + 1, len(factors)):
                    fig, ax = plt.subplots(figsize=(6, 4))
                    sns.pointplot(data=df, x=factors[i], y=response, hue=factors[j], ax=ax)
                    ax.set_title(f"Interaction Effect: {factors[i]} x {factors[j]}")
                    st.pyplot(fig)

            # Effect Strength Interpretation
            st.header("Effect Strength Interpretation")
            effect_sizes = model.params[1:]  # Exclude the intercept
            descriptions = []
            for effect, size in effect_sizes.items():
                if abs(size) > 1.0:
                    descriptions.append(f"{effect}: Strong relationship")
                elif abs(size) > 0.5:
                    descriptions.append(f"{effect}: Moderate relationship")
                else:
                    descriptions.append(f"{effect}: Weak relationship")
            for description in descriptions:
                st.write(description)
            
            # Blocking Chart
            st.header("Boxplots")
            if 'Replicate' in df.columns:
                fig, ax = plt.subplots(figsize=(6, 4))
                sns.boxplot(data=df, x='Replicate', y=response, ax=ax)
                ax.set_title("Blocking Chart by Replicate")
                st.pyplot(fig)
            else:
                st.write("No blocking factor found in the data.")

            # Pareto Chart of Effects
            st.header("Pareto Chart of Effects")
            
            # Sort the effects by absolute value in descending order
            effect_sizes_sorted = effect_sizes.reindex(effect_sizes.abs().sort_values(ascending=False).index)
            
            # Pareto Chart
            fig, ax = plt.subplots(figsize=(8, 6))
            bars = ax.barh(effect_sizes_sorted.index, effect_sizes_sorted.abs(), color="skyblue")
            ax.set_xlabel("Absolute Effect Size")
            ax.set_title("Pareto Chart of Factor Effects")

            # Highlight significant factors
            significance_threshold = 1.0  # Example threshold for strong effect
            for bar, effect in zip(bars, effect_sizes_sorted):
                if abs(effect) > significance_threshold:
                    bar.set_color("orange")

            st.pyplot(fig)

            # Residuals Analysis
            st.header("Residual Analysis")
            
            # Calculate residuals
            residuals = model.resid
            fitted_values = model.fittedvalues
            order = np.arange(len(residuals)) + 1

            # Normal Probability Plot of Residuals
            st.subheader("Normal Probability Plot (Residuals)")
            fig, ax = plt.subplots()
            sm.qqplot(residuals, line='45', ax=ax)
            ax.set_title("Normal Probability Plot of Residuals")
            st.pyplot(fig)

            # Dynamic interpretation of Normal Probability Plot
            if abs(residuals.skew()) < 0.5:
                st.write("Interpretation: The residuals appear to follow a normal distribution, as they closely align with the 45-degree line. This suggests that the model's errors are normally distributed.")
            else:
                st.write("Interpretation: The residuals deviate from the 45-degree line, indicating a non-normal distribution. This may suggest issues with model assumptions or data outliers.")

            # Residuals vs Fitted Values
            st.subheader("Residuals vs Fitted Values")
            fig, ax = plt.subplots()
            ax.scatter(fitted_values, residuals)
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel("Fitted Values")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Fitted Values")
            st.pyplot(fig)

            # Dynamic interpretation of Residuals vs Fitted Values
            if residuals.var() < 0.05 * np.mean(fitted_values):
                st.write("Interpretation: The residuals appear randomly scattered around zero, which suggests homoscedasticity (constant variance) and a good model fit.")
            else:
                st.write("Interpretation: The residuals show a pattern or non-random spread, which indicates heteroscedasticity (non-constant variance). This suggests that the model's accuracy may vary across levels of the fitted values.")

            # Histogram of Residuals
            st.subheader("Histogram of Residuals")
            fig, ax = plt.subplots()
            sns.histplot(residuals, kde=True, ax=ax)
            ax.set_title("Histogram of Residuals")
            st.pyplot(fig)

            # Dynamic interpretation of Histogram of Residuals
            if abs(residuals.skew()) < 0.5:
                st.write("Interpretation: The histogram shows a roughly symmetric distribution, supporting the assumption of normality in the residuals.")
            else:
                st.write("Interpretation: The histogram is skewed, indicating that the residuals may not be normally distributed, potentially affecting model reliability.")

            # Residuals vs Observation Order
            st.subheader("Residuals vs Observation Order")
            fig, ax = plt.subplots()
            ax.plot(order, residuals, marker='o')
            ax.axhline(0, color='red', linestyle='--')
            ax.set_xlabel("Observation Order")
            ax.set_ylabel("Residuals")
            ax.set_title("Residuals vs Observation Order")
            st.pyplot(fig)

            # Dynamic interpretation of Residuals vs Observation Order
            if np.var(np.diff(residuals)) < 0.05 * np.var(residuals):
                st.write("Interpretation: The residuals do not show a noticeable trend or pattern over time, suggesting that there are no time-dependent patterns in the errors.")
            else:
                st.write("Interpretation: The residuals display a trend or pattern over time, which may suggest that certain factors were not adequately controlled, or that there is a potential time-based effect in the data.")

        else:
            st.warning("Please select at least one factor for analysis.")

# Response Optimizer
if tool_selection == "Response Optimizer":
    
    st.header("Response Optimizer")
    # Step 1: Upload CSV file
    st.header("1. Upload CSV File")
    uploaded_file = st.file_uploader("Upload your DOE data file", type="csv")

    if uploaded_file:
        # Load data
        df = pd.read_csv(uploaded_file)
        st.write("Uploaded Data:", df)

        # Step 2: Select Response Variable and Optimization Goals
        st.header("2. Set Response and Goals")
        
        response = st.selectbox("Select the Response Variable:", options=df.columns)
        factors = st.multiselect("Select Factors for Optimization:", options=[col for col in df.columns if col != response])
        
        if factors:
            goal = st.selectbox("Goal for Response:", ["Minimize", "Maximize"])
            target = st.number_input("Target Value (Optional)", value=np.nan)
            min_bound = st.number_input("Minimum Bound", value=float(df[response].min()))
            max_bound = st.number_input("Maximum Bound", value=float(df[response].max()))

            # Step 3: Regression Model and Algebraic Formula Display
            st.header("3. Regression Model (Algebraic Formula)")
            
            # Create formula for regression with main effects and interactions
            formula = f"{response} ~ " + " + ".join(factors) + " + " + " + ".join([f"{i}:{j}" for i in factors for j in factors if i != j])
            model = smf.ols(formula, data=df).fit()
            
            # Extract coefficients to create an algebraic formula
            coefficients = model.params
            equation = f"{response} = "
            for i, (term, coef) in enumerate(coefficients.items()):
                if i == 0:
                    equation += f"{coef:.4f}"  # Intercept term
                else:
                    equation += f" + ({coef:.4f} * {term})"

            # Display the simplified algebraic formula
            st.write("Regression Formula (Algebraic Form):")
            st.latex(equation)

            # Step 4: Optimization based on Factor Levels
            st.header("4. Optimization")

            # Define objective function for optimization
            def objective(x):
                factor_dict = {factor: x[i] for i, factor in enumerate(factors)}
                y_pred = model.predict(pd.DataFrame([factor_dict]))[0]
                if goal == "Minimize":
                    return y_pred if np.isnan(target) else abs(y_pred - target)
                elif goal == "Maximize":
                    return -y_pred if np.isnan(target) else abs(y_pred - target)

            # Set bounds for each factor based on data range
            bounds = [(df[f].min(), df[f].max()) for f in factors]

            # Perform optimization
            initial_guess = [(df[f].mean()) for f in factors]
            result = opt.minimize(objective, initial_guess, bounds=bounds)

            if result.success:
                optimal_values = result.x
                st.write("Optimization Successful!")
                st.write("Optimal Factor Levels:")
                for factor, value in zip(factors, optimal_values):
                    st.write(f"{factor}: {value:.4f}")
                
                # Predicted response at optimal factor levels
                optimal_dict = {factor: optimal_values[i] for i, factor in enumerate(factors)}
                predicted_response = model.predict(pd.DataFrame([optimal_dict]))[0]
                st.write(f"Predicted {response} at Optimal Levels: {predicted_response:.4f}")

                # Goal achievement summary
                if not np.isnan(target):
                    if goal == "Minimize" and predicted_response <= target:
                        st.write(f"The optimization meets the target to minimize the {response} to {target}.")
                    elif goal == "Maximize" and predicted_response >= target:
                        st.write(f"The optimization meets the target to maximize the {response} to {target}.")
                    else:
                        st.write(f"The optimization does not meet the target. Predicted value is {predicted_response}, with a target of {target}.")
                else:
                    st.write(f"Optimization achieved a {goal.lower()}d {response} of {predicted_response}.")
                    
                st.write("___________________________________")
                # Step 5: Response Prediction
                st.header("Response Prediction and Factor Simulation")

                # Allow user to input levels for each factor to predict the response
                st.subheader("Enter Factor Levels for Prediction")
                factor_levels = {}
                for factor in factors:
                    level = st.number_input(f"Enter level for {factor}:", value=float(df[factor].mean()))
                    factor_levels[factor] = level

                # Predict the response based on input factor levels
                if st.button("Predict Response"):
                    # Convert input to DataFrame for model prediction
                    input_df = pd.DataFrame([factor_levels])
                    
                    # Predict the response using the regression model
                    predicted_response = model.predict(input_df)[0]
                    
                    st.write(f"Predicted {response}: {predicted_response:.4f}")

            else:
                st.write("Optimization was unsuccessful. Try adjusting bounds or checking data.")
        
        else:
            st.warning("Please select at least one factor for optimization.")
            
            
with open("style.css") as f:
    css = f.read()
    
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)
