import streamlit as st
import pickle, sklearn
import numpy as np
import pandas as pd
import tensorflow as tf


st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📉",
    layout="centered",
    initial_sidebar_state="auto",
    menu_items={
        'Get Help': 'https://bit.ly/naufal-git',
        'Report a bug': "https://bit.ly/naufal-linkedin",
        'About': "This web app is made as part of Hacktiv8 Full Time Data Science program.",
    }
)

st.title('Churn Prediction with ANN')
#st.image('header.jpg')

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('ohe.pkl', 'rb') as f:
    ohe = pickle.load(f)

with open('pca18.pkl', 'rb') as f:
    pca = pickle.load(f)

@st.cache(allow_output_mutation=True)
def teachable_machine_classification(new_data, weights_file):
    # Load the model
    #model = tf.saved_model.load(weights_file)
    model = tf.keras.models.load_model(weights_file)
    
    num_columns = new_data.select_dtypes(include=np.number).columns.tolist()
    cat_columns = new_data.select_dtypes(include=['object']).columns.tolist()

    # Split between Numerical Columns and Categorical Columns
    data_inf_num = new_data[num_columns]
    data_inf_cat = new_data[cat_columns]

    data_inf_num_scaled = scaler.transform(data_inf_num)
    data_inf_num_scaled = pd.DataFrame(data_inf_num_scaled, columns=data_inf_num.columns)

    data_inf_cat_ohe = ohe.transform(data_inf_cat)
    data_inf_cat_ohe = pd.DataFrame(data_inf_cat_ohe)
    #data_inf_cat_ohe.columns = ohe.get_feature_names_out(input_features=data_inf_cat.columns)

    data_inf_final = pd.concat([data_inf_num_scaled, data_inf_cat_ohe], axis=1)

    data_inf_pca = pca.transform(data_inf_final)
    

    # run the inference
    predictions = model.predict(data_inf_pca)
    predictions = np.where(predictions >= 0.5,1,0)

    return  predictions


gender_in = st.selectbox("Please specify customer's gender", ['male', 'female'])
SeniorCitizen_in = st.selectbox("Is the customer 65 years old or older? (0 = No, 1 = Yes)", [0, 1])
Partner_in = st.selectbox("Does the customer have a partner?", ['Yes', 'No'])
Dependents_in = st.selectbox("Does the customer have any dependent?", ['Yes', 'No'])
Tenure_in = st.number_input("How many months has the customer been a customer?", min_value=0, max_value=72, value=0)
PhoneService_in = st.selectbox("Does the customer subscribe to home phone service?", ['Yes', 'No'])
MultipleLines_in = st.selectbox("Does the customer subscribe to multiple telephone lines?", ['Yes', 'No', 'No phone service'])
InternetService_in = st.selectbox("Does the customer subscribe to internet service?", ['DSL', 'Fiber optic', 'No'])
OnlineSecurity_in = st.selectbox("Does the customer subscribe to the company's additional online security service?", ['Yes', 'No'])
OnlineBackup_in = st.selectbox("Does the customer subscribe to additional online backup service?", ['Yes', 'No'])
DeviceProtection_in = st.selectbox("Does the customer subscribe to additional device protection service?", ['Yes', 'No'])
TechSupport_in = st.selectbox("Does the customer subscribe to additional technical support plan?", ['Yes', 'No'])
StreamingTV_in = st.selectbox("Does the customer use their internet to stream TV programs from a third party provider?", ['Yes', 'No'])
StreamingMovies_in = st.selectbox("Does the customer use their internet to stream movies from a third party provider?", ['Yes', 'No'])
Contract_in = st.selectbox("Please specify the customer's current contract!", ['Month-to-Month', 'One Year', 'Two Year'])
PaperlessBilling_in = st.selectbox("Does the customer have chosen paperless billing?", ['Yes', 'No'])
PaymentMethod_in = st.selectbox("Please specify the customer's payment method!", ['Bank Withdrawal', 'Credit Card', 'Mailed Check'])
MonthlyCharges_in = st.number_input("What is the customer's current monthly charges?", min_value=0, max_value=200, value=0)
TotalCharges_in = st.number_input("What is the customer's total charges? Calculated to the end of the quarter", min_value=0, max_value=1000, value=0)


# inference
data_inf = {'gender':gender_in,
        'SeniorCitizen':SeniorCitizen_in,
        'Partner':Partner_in,
        "Dependents":Dependents_in,
        'tenure':Tenure_in,
        'PhoneService':PhoneService_in,
        'MultipleLines':MultipleLines_in,
        'InternetService':InternetService_in,
        'OnlineSecurity':OnlineSecurity_in,
        'OnlineBackup':OnlineBackup_in,
        'DeviceProtection':DeviceProtection_in,
        'TechSupport':TechSupport_in,
        'StreamingTV':StreamingTV_in,
        'StreamingMovies':StreamingMovies_in,
        'Contract':Contract_in,
        'PaperlessBilling':PaperlessBilling_in,
        'PaymentMethod':PaymentMethod_in,
        'MonthlyCharges':MonthlyCharges_in,
        'TotalCharges':TotalCharges_in
        }

columns = [ 'gender', 'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 
            'PhoneService', 'MultipleLines', 'InternetService', 'OnlineSecurity',
            'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV',
            'StreamingMovies', 'Contract', 'PaperlessBilling', 'PaymentMethod',
            'MonthlyCharges', 'TotalCharges']

new_df = pd.DataFrame([data_inf], columns=columns)

# URL = 'https://nau-churn-be.herokuapp.com/churn'

# r = requests.post(URL, json=data_inf)


# if st.button('Predict'):
#     st.write("Predicting...")
#     res = r.json()['prediction']
#     if res == 1:
#         st.write('The customer is likely to leave the company')
#     elif res == 0:
#         st.write('The customer is likely to stay in the company')
#     else:
#         st.write('Something went wrong')
# else:
#     st.write("Please click the button to predict")

if st.button('Predict'):
    st.write("Predicting...")
    res = teachable_machine_classification(new_df, 'tuned_model.h5')
    if res == 1:
        st.write('The customer is likely to leave the company')
    elif res == 0:
        st.write('The customer is likely to stay in the company')
    else:
        st.write('Something went wrong')
else:
    st.write("Please click the button to predict")