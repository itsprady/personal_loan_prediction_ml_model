import streamlit as st
import pandas as pd
import numpy as np
from joblib import load
from sklearn.preprocessing import StandardScaler

# Load the saved model and scaler
@st.cache_resource
def load_model():
    model = load('compressed_loan_model.pbz2')
    scaler = load('scaler.joblib')
    return model, scaler

model, scaler = load_model()

# Extract feature names from the trained model
trained_features = model.feature_names_in_
print("Trained Features:", trained_features)

# List of expected features (from your training data)
expected_features = [
    'Customer_Age', 'Dependent_Count', 'Income', 'Cust_Satisfaction_Score',
    'Car_Owner', 'House_Owner', 'Gender_M', 'Education_Level_Graduate',
    'Education_Level_High School', 'Education_Level_Post-Graduate',
    'Education_Level_Uneducated', 'Education_Level_Unknown',
    'Marital_Status_Married', 'Marital_Status_Single', 'Marital_Status_Unknown',
    'state_cd_CA', 'state_cd_FL', 'state_cd_MA', 'state_cd_MO', 'state_cd_NJ',
    'state_cd_NY', 'state_cd_TX', 'state_cd_AR', 'state_cd_AZ', 'state_cd_CO',
    'contact_telephone', 'contact_unknown', 'Customer_Job_Blue-collar',
    'Customer_Job_Businessman', 'Customer_Job_Govt', 'Customer_Job_Retirees',
    'Customer_Job_Selfemployeed', 'Customer_Job_White-collar'
]

# Function to preprocess input data
def preprocess_data(input_df):
    # Convert binary columns
    input_df['Car_Owner'] = input_df['Car_Owner'].map({'yes': 1, 'no': 0})
    input_df['House_Owner'] = input_df['House_Owner'].map({'yes': 1, 'no': 0})
    
    # One-hot encoding for categorical variables
    categorical_cols = {
        'Gender': ['M'],
        'Education_Level': ['Graduate', 'High School', 'Post-Graduate', 'Uneducated', 'Unknown'],
        'Marital_Status': ['Married', 'Single', 'Unknown'],
        'state_cd': ['CA', 'FL', 'MA', 'MO', 'NJ', 'NY', 'TX', 'AR', 'AZ', 'CO', 'CT', 'GA', 'HI', 'IA', 'IL'],
        'contact': ['telephone', 'unknown'],
        'Customer_Job': ['Blue-collar', 'Businessman', 'Govt', 'Retirees', 'Selfemployeed', 'White-collar']
    }
    
    for col, values in categorical_cols.items():
        for val in values:
            input_df[f"{col}_{val}"] = (input_df[col] == val).astype(int)
    
    # Ensure all expected columns are present
    for feature in trained_features:  # Use model's trained features
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Remove extra columns not in trained_features
    input_df = input_df[trained_features]
    
    # Scale numerical features
    num_cols = ['Customer_Age', 'Dependent_Count', 'Income', 'Cust_Satisfaction_Score']
    input_df[num_cols] = scaler.transform(input_df[num_cols])
    
    return input_df

# Function to make predictions
def make_prediction(data):
    processed_data = preprocess_data(data)
    prediction = model.predict(processed_data)
    probability = model.predict_proba(processed_data)[:, 1]
    return prediction, probability

# Streamlit app layout
st.title("Personal Loan Approval Predictor")
st.write("Predict whether a customer will accept a personal loan offer")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["Single Prediction", "Batch Prediction"])

with tab1:
    st.header("Enter Customer Details")
    
    with st.form("single_prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=100, value=30)
            dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
            income = st.number_input("Annual Income", min_value=0, value=50000)
            satisfaction = st.slider("Customer Satisfaction (1-3)", 1, 3, 2)
            
        with col2:
            gender = st.selectbox("Gender", ["F", "M"])
            education = st.selectbox("Education Level", 
                                   ["Uneducated", "High School", "Graduate", "Post-Graduate", "Unknown"])
            marital_status = st.selectbox("Marital Status", ["Single", "Married", "Unknown"])
            state = st.selectbox("State", ["CA", "FL", "MA", "MO", "NJ", "NY", "TX", "AR", "AZ", "CO"])
            job = st.selectbox("Occupation", 
                             ["Businessman", "Blue-collar", "Govt", "Retirees", "Selfemployeed", "White-collar"])
            car_owner = st.radio("Owns a Car?", ["yes", "no"])
            house_owner = st.radio("Owns a House?", ["yes", "no"])
            contact = st.selectbox("Preferred Contact Method", ["cellular", "telephone", "unknown"])
        
        submitted = st.form_submit_button("Predict")
        
        if submitted:
            # Create dataframe from inputs
            input_data = pd.DataFrame([{
                'Customer_Age': age,
                'Dependent_Count': dependents,
                'Income': income,
                'Cust_Satisfaction_Score': satisfaction,
                'Gender': gender,
                'Education_Level': education,
                'Marital_Status': marital_status,
                'state_cd': state,
                'Customer_Job': job,
                'Car_Owner': car_owner,
                'House_Owner': house_owner,
                'contact': contact
            }])
            
            # Make prediction
            prediction, probability = make_prediction(input_data)
            
            # Display results
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.success(f"Loan Approval Prediction: **Approved** (Probability: {probability[0]:.2%})")
            else:
                st.error(f"Loan Approval Prediction: **Rejected** (Probability: {probability[0]:.2%})")

with tab2:
    st.header("Upload CSV File for Batch Prediction")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            # Read the uploaded file
            batch_data = pd.read_csv(uploaded_file)
            
            # Show preview
            st.write("Preview of uploaded data:")
            st.dataframe(batch_data.head())
            
            if st.button("Predict for All"):
                # Make predictions
                predictions, probabilities = make_prediction(batch_data)
                
                # Add results to original data
                results_df = batch_data.copy()
                results_df['Loan_Prediction'] = ['Approved' if p == 1 else 'Rejected' for p in predictions]
                results_df['Approval_Probability'] = probabilities
                
                # Show results
                st.subheader("Prediction Results")
                st.dataframe(results_df)
                
                # Download button
                csv = results_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="Download Predictions as CSV",
                    data=csv,
                    file_name='loan_predictions.csv',
                    mime='text/csv'
                )
                
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# Add this after your existing tabs

tab3 = st.tabs(["Data Insights"])[0]
with tab3:
    st.header("Exploratory Data Analysis & Insights")

    # Load and merge data (adjust file paths as needed)
    @st.cache_data
    def load_data():
        df_cust = pd.read_csv("customer.csv")
        df_card = pd.read_csv("credit_card.csv")
        df = pd.merge(df_cust, df_card, on="Client_Num", how="inner")
        df.columns = df.columns.str.strip()  # Remove extra spaces
        return df

    df = load_data()

    # 1. Correlation heatmap for customer satisfaction
    st.subheader("1. Correlation Heatmap")
    import seaborn as sns
    import matplotlib.pyplot as plt
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
    st.pyplot(fig)

    # 1. Boxplot of satisfaction by education
    st.subheader("Satisfaction by Education Level")
    fig, ax = plt.subplots()
    sns.boxplot(x='Education_Level', y='Cust_Satisfaction_Score', data=df, ax=ax)
    st.pyplot(fig)

    # 2. Grouped bar: Credit card usage by gender
    st.subheader("Total Transaction Amount by Gender")
    fig, ax = plt.subplots()
    sns.barplot(x='Gender', y='Total_Trans_Amt', data=df, ci=None, ax=ax)
    st.pyplot(fig)

    # 2. Violin plot: Usage by age group
    st.subheader("Transaction Amount by Age Group")
    df['AgeGroup'] = pd.cut(df['Customer_Age'], bins=[18,30,40,50,60,100])
    fig, ax = plt.subplots()
    sns.violinplot(x='AgeGroup', y='Total_Trans_Amt', data=df, ax=ax)
    st.pyplot(fig)

    # 3. Scatter: Credit limit vs delinquency
    st.subheader("Credit Limit vs Delinquency")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Credit_Limit', y='Delinquent_Acc', data=df, alpha=0.6, ax=ax)
    st.pyplot(fig)

    # 3. Boxplot: Credit limit by delinquency
    st.subheader("Credit Limit by Delinquency Status")
    fig, ax = plt.subplots()
    sns.boxplot(x='Delinquent_Acc', y='Credit_Limit', data=df, ax=ax)
    st.pyplot(fig)

    # 4. Bar: Average transaction by state
    st.subheader("Average Transaction Amount by State")
    state_avg = df.groupby('state_cd')['Total_Trans_Amt'].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12,6))
    state_avg.plot(kind='bar', ax=ax)
    st.pyplot(fig)

    # 5. Boxplot: Utilization ratio by job type
    st.subheader("Utilization Ratio by Job Type")
    fig, ax = plt.subplots(figsize=(12,6))
    sns.boxplot(x='Customer_Job', y='Avg_Utilization_Ratio', data=df, ax=ax)
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # 6. Line: Activation rate over weeks
    st.subheader("Activation Rate Over Weeks")
    activation_rate = df.groupby('Week_Num')['Activation_30_Days'].mean()
    fig, ax = plt.subplots()
    activation_rate.plot(kind='line', marker='o', ax=ax)
    st.pyplot(fig)

    # 7. Stacked bar: Expense type by marital status
    st.subheader("Expense Type by Marital Status")
    exp_type = pd.crosstab(df['Exp Type'], df['Marital_Status'])
    fig, ax = plt.subplots()
    exp_type.plot(kind='bar', stacked=True, ax=ax)
    st.pyplot(fig)

    # 8. Scatter: Income vs Total Transaction Amount colored by satisfaction
    st.subheader("Income vs Total Transaction Amount (colored by Satisfaction)")
    fig, ax = plt.subplots()
    sns.scatterplot(x='Income', y='Total_Trans_Amt', hue='Cust_Satisfaction_Score', data=df, palette='viridis', ax=ax)
    st.pyplot(fig)

    # 9. Bar: Personal loan acceptance by education
    st.subheader("Personal Loan Acceptance by Education Level")
    fig, ax = plt.subplots()
    sns.countplot(x='Education_Level', hue='Personal_loan', data=df, ax=ax)
    st.pyplot(fig)

    # 10. Feature importance (if you have a trained model)
    st.subheader("Feature Importance (Model-Based)")
    try:
        importances = model.feature_importances_
        feat_names = model.feature_names_in_
        feat_imp = pd.Series(importances, index=feat_names).sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(12,6))
        feat_imp.plot(kind='bar', ax=ax)
        st.pyplot(fig)
    except Exception as e:
        st.info("Feature importance plot not available for this model type.")

    st.info("Scroll to see all insights. You can interact with the app using the tabs above.")

# Add some instructions
st.sidebar.markdown("""
### Instructions
- **Single Prediction**: Fill out the form with customer details
- **Batch Prediction**: Upload a CSV file with customer data
- Required columns: All fields shown in the single prediction form

### Expected CSV Format
Your CSV should include these columns (case sensitive):
""")
