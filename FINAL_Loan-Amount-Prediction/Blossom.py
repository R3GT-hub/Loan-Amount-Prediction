import pickle
import streamlit as st
import numpy as np

st.title('Loan Prediction Assistant')

st.markdown('This application helps banks determine the loan amount a user may qualify for based on several factors.')
st.markdown('Please enter the details below to see the prediction.')

# loading the trained model
try:
    with open('blossom2.pkl', 'rb') as pickle_in:
        classifier = pickle.load(pickle_in)
except Exception as e:
    st.error(f"Error loading model: {e}")

@st.cache_data()

def convert_ghc_to_inr(ghc_amount):
    conversion_rate = 13.5  # Example conversion rate, 1 GHC = 13.5 INR
    return ghc_amount * conversion_rate
def convert_inr_to_ghc(inr_amount):
    conversion_rate = 0.074  # Example conversion rate, 1 INR = 0.074 GHC
    return inr_amount * conversion_rate


# Define prediction function using one-hot encoding for categorical features
def prediction(Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome,
               CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area):   

    # One-hot encode categorical input
    gender_female = 1 if Gender == 'Female' else 0
    gender_male = 1 - gender_female
    married_yes = 1 if Married == 'Yes' else 0
    married_no = 1 - married_yes

    dependents_0 = 1 if Dependents == '0' else 0
    dependents_1 = 1 if Dependents == '1' else 0
    dependents_2 = 1 if Dependents == '2' else 0
    dependents_3plus = 1 if Dependents == '3+' else 0

    education_graduate = 0 if Education == 'Graduate' else 1
    self_employed = 1 if Self_Employed == 'Yes' else 0
    property_rural = 1 if Property_Area == 'Rural' else 0
    property_semiurban = 1 if Property_Area == 'Semiurban' else 0
    property_urban = 1 - property_rural - property_semiurban
    

    # Construct input array based on one-hot encoding
    features = [gender_female, gender_male, married_no, married_yes,
                dependents_0, dependents_1, dependents_2, dependents_3plus,
                education_graduate, self_employed, ApplicantIncome, CoapplicantIncome,
                LoanAmount, Loan_Amount_Term, Credit_History,
                property_rural, property_semiurban, property_urban]

    # Make prediction
    prediction = classifier.predict([features])[0]
    return prediction

# main function to define webpage layout and input fields
def main():
    html_temp = """ 
    <div style="background-color:yellow;padding:13px"> 
    <h1 style="color:black;text-align:center;">Loan Prediction ML App</h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Input fields for user data
    Gender = st.selectbox("Gender", ('Male', 'Female'))
    Married = st.selectbox("Are You Married", ('Yes', 'No'))
    Dependents = st.selectbox("How Many Dependents", ('0', '1', '2', '3+'))
    Education = st.selectbox("Education Level", ('Graduate', 'Non Graduate'))
    Self_Employed = st.selectbox("Are you Self Employed?", ('Yes', 'No'))
    ApplicantIncome = st.number_input("Applicant's Income (In INR)", min_value=0)
    CoapplicantIncome = st.number_input("Co-Applicant's Income (In INR)", min_value=0)
    LoanAmount = st.number_input("Loan Amount (In INR)", min_value=0)
    Loan_Amount_Term = st.number_input("Loan Amount Term (In Days)", min_value=1)
    Credit_History = st.selectbox("Credit History (1=Yes, 0=No)", ('1', '0'))
    Property_Area = st.selectbox("Property Location", ('Rural', 'Urban', 'Semiurban'))

     # Convert INR to GHC
    ApplicantIncome = convert_inr_to_ghc(ApplicantIncome)
    CoapplicantIncome = convert_inr_to_ghc(CoapplicantIncome)
    LoanAmount = convert_inr_to_ghc(LoanAmount)


    # Button to make prediction
    if st.button("Predict Loan Amount"):
        try:
            output = prediction(Gender, Married, Dependents, Education, Self_Employed,
                                ApplicantIncome, CoapplicantIncome, LoanAmount,
                                Loan_Amount_Term, int(Credit_History), Property_Area)
            output = convert_ghc_to_inr(output)
            st.success(f"Predicted Loan Amount: {output} INR 🙌")
        except Exception as e:
            st.warning(f"Oops! Something went wrong: {e}")

if __name__ == '__main__': 
    main()
