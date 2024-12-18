# ======================================================
# Import Libraries
# ======================================================
import pandas as pd
import streamlit as st
import pickle

# ======================================================
# Title
# ======================================================
st.title("Customer Lifetime Value Predictor")
st.write("Predict the lifetime value of a customer and gain insights into key contributing features.")

# Sidebar Header
st.sidebar.header("Input Customer Features")

# ======================================================
# Function to Collect User Input
# ======================================================
def user_input_features():
    # Numerical Inputs
    Monthly_Premium_Auto = st.sidebar.number_input('Monthly Premium Auto ($)', min_value=0, value=100, step=1)
    Log_Total_Claim_Amount = st.sidebar.number_input('Log Total Claim Amount', min_value=0.0, value=5.0, step=0.1)
    Income = st.sidebar.number_input('Income ($)', min_value=0, value=50000, step=1000)
    Number_of_Policies = st.sidebar.number_input('Number of Policies', min_value=1, value=2, step=1)

    # Categorical Inputs
    Vehicle_Class = st.sidebar.selectbox('Vehicle Class', ['Luxury', 'SUV', 'Sedan'])
    Coverage = st.sidebar.selectbox('Coverage', ['Basic', 'Extended', 'Premium'])
    Renew_Offer_Type = st.sidebar.selectbox('Renew Offer Type', ['Offer1', 'Offer2', 'Offer3', 'Offer4'])
    Employment_Status = st.sidebar.selectbox('Employment Status', ['Employed', 'Unemployed', 'Student', 'Retired'])
    Marital_Status = st.sidebar.selectbox('Marital Status', ['Single', 'Married', 'Divorced'])
    Education = st.sidebar.selectbox('Education', ['High School', 'Bachelor', 'Master', 'PhD'])

    # Create a DataFrame
    data = {
        'Monthly_Premium_Auto': [Monthly_Premium_Auto],
        'Log_Total_Claim_Amount': [Log_Total_Claim_Amount],
        'Income': [Income],
        'Number_of_Policies': [Number_of_Policies],
        'Vehicle_Class': [Vehicle_Class],
        'Coverage': [Coverage],
        'Renew_Offer_Type': [Renew_Offer_Type],
        'Employment_Status': [Employment_Status],
        'Marital_Status': [Marital_Status],
        'Education': [Education]
    }

    return pd.DataFrame(data)

# Collect User Input
df_customer = user_input_features()

# Encode Categorical Features
categorical_features = ['Vehicle_Class', 'Coverage', 'Renew_Offer_Type', 
                        'Employment_Status', 'Marital_Status', 'Education']
df_customer_encoded = pd.get_dummies(df_customer, columns=categorical_features)

# Ensure the input DataFrame matches the model's expected format
expected_columns = [
    'Monthly_Premium_Auto', 'Log_Total_Claim_Amount', 'Income', 
    'Number_of_Policies', 'Vehicle_Class_Luxury', 'Vehicle_Class_SUV', 
    'Vehicle_Class_Sedan', 'Coverage_Basic', 'Coverage_Extended', 'Coverage_Premium', 
    'Renew_Offer_Type_Offer1', 'Renew_Offer_Type_Offer2', 'Renew_Offer_Type_Offer3', 
    'Renew_Offer_Type_Offer4', 'Employment_Status_Employed', 'Employment_Status_Retired', 
    'Employment_Status_Student', 'Employment_Status_Unemployed', 'Marital_Status_Divorced', 
    'Marital_Status_Married', 'Marital_Status_Single', 'Education_Bachelor', 
    'Education_High School', 'Education_Master', 'Education_PhD'
]

# Add missing columns with zeros
for col in expected_columns:
    if col not in df_customer_encoded.columns:
        df_customer_encoded[col] = 0

# Reorder columns to match the expected model input
df_customer_encoded = df_customer_encoded[expected_columns]

# ======================================================
# Load Model and Make Prediction
# ======================================================
try:
    # Load the trained model
    model = pickle.load(open('garnet.pkl', 'rb'))
    predicted_clv = model.predict(df_customer_encoded)[0]
except FileNotFoundError:
    st.error("Model file 'garnet.pkl' not found. Please ensure the model is in the same directory.")
    predicted_clv = None
except Exception as e:
    st.error(f"An error occurred during prediction: {e}")
    predicted_clv = None

# ======================================================
# Display Results
# ======================================================
col1, col2 = st.columns(2)

# Left Column: Input Features
with col1:
    st.subheader("Customer Features:")
    st.write(df_customer.transpose())

# Right Column: Prediction and Feature Importance
with col2:
    st.subheader("Predicted CLV:")
    if predicted_clv is not None:
        st.write(f"${predicted_clv:,.2f}")

    # Feature importance visualization (if supported by the model)
    if hasattr(model, 'feature_importances_'):
        feature_importances = pd.DataFrame({
            'Feature': expected_columns,
            'Importance': model.feature_importances_
        }).sort_values(by='Importance', ascending=False)

        st.subheader("Feature Importance:")
        st.bar_chart(feature_importances.set_index('Feature'))

# ======================================================
# Footer
# ======================================================
st.write("Developed as part of the Customer Lifetime Value Project")
