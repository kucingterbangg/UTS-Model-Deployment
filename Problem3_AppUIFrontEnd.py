# link video = https://drive.google.com/drive/folders/136ABi3Ju191lqCTVlzZG7UYQqJPCLw5P?usp=sharing

import streamlit as st
import joblib
import numpy as np
import pandas as pd

def main():
    st.title('Machine Learning Loan Status Prediction')
    st.subheader('To predict the approval of a loan')

    # Load model and get expected feature order
    try:
        model = joblib.load('XGB_class.pkl')
        expected_features = model.get_booster().feature_names
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return

    with st.form("loan_input_form"):
        st.header("Biodata")
        col1, col2 = st.columns(2)
        with col1:
            #min max value follows df.describe() function (statistical summary)
            person_age = st.number_input('Age', min_value=20, max_value=144, value=20)
            person_gender = st.radio('Gender', ['male', 'female'])
            person_education = st.selectbox('Education Level', [
                'High School', 'Associate', 'Bachelor', 'Master', 'Doctorate'
            ])
            
        with col2:
            person_income = st.number_input('Annual Income', min_value=0, value=0)
            person_emp_exp = st.number_input('Employment Experience (years)', min_value=0, max_value=125, value=5)
            person_home_ownership = st.selectbox('Home Ownership', [
                'RENT', 'OWN', 'MORTGAGE', 'OTHER'
            ])
        
        st.header("Loan Details")
        col3, col4 = st.columns(2)
        with col3:
            loan_amnt = st.number_input('Loan Amount', min_value=500, max_value=35000, value=500)
            loan_intent = st.selectbox('Loan Purpose', [
                'EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 
                'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT'
            ])
            loan_int_rate = st.number_input('Interest Rate (%)', min_value=5.42, max_value=20.0, value=5.42)
            
        with col4:
            loan_percent_income = st.slider('Loan Percentage of Income', min_value=0.0, max_value=0.66, value=0.0)
            cred_hist_length = st.number_input('Credit History Length (years)', min_value=2, max_value=30, value=2)
            previous_defaults = st.radio('Previous Loan Defaults', ['No', 'Yes'])
        
        credit_score = st.slider('Credit Score', min_value=390, max_value=850, value=390)
        
        submitted = st.form_submit_button("Predict Loan Status")
    
    if submitted:
        # Prepare input data with consistent feature order
        input_data = {
            'person_age': person_age,
            'person_income': person_income,
            'person_emp_exp': person_emp_exp,
            'person_gender': 1 if person_gender == 'male' else 0,
            'previous_loan_defaults_on_file': 1 if previous_defaults == 'Yes' else 0,
            'loan_amnt': loan_amnt,
            'loan_int_rate': loan_int_rate,
            'loan_percent_income': loan_percent_income,
            'cb_person_cred_hist_length': cred_hist_length,
            'credit_score': credit_score
        }
        
        # One-hot encoding - must match training exactly
        for intent in ['EDUCATION', 'MEDICAL', 'VENTURE', 'PERSONAL', 'DEBTCONSOLIDATION', 'HOMEIMPROVEMENT']:
            input_data[f'loan_intent_{intent}'] = 1 if loan_intent == intent else 0
        
        for ownership in ['RENT', 'OWN', 'MORTGAGE', 'OTHER']:
            input_data[f'person_home_ownership_{ownership}'] = 1 if person_home_ownership == ownership else 0
        
        for education in ['High School', 'Associate', 'Bachelor', 'Master', 'Doctorate']:
            input_data[f'person_education_{education}'] = 1 if person_education == education else 0
        
        # Create DataFrame with columns in EXACTLY the expected order
        input_df = pd.DataFrame([input_data])[expected_features]
        
        try:
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)
            
            st.subheader("Prediction Result")
            if prediction[0] == 1:
                st.success('Loan Approved')
            else:
                st.error('Loan is not Approved')
            
            st.write(f"Confidence: {np.max(probability)*100:.2f}%")
            
            prob_df = pd.DataFrame({
                'Status': ['Approved', 'Not Approved'],
                'Probability': [probability[0][1], probability[0][0]]
            })
            st.bar_chart(prob_df.set_index('Status'))
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")

if __name__ == '__main__':
    main()