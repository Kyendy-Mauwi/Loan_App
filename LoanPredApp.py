import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle 

# Load the trained model
# model = joblib.load("my_model1.pkl")
with open("my_model1.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Loan Repayment Prediction App")

st.sidebar.header("User Input")

# Get user input
initial_payment = st.sidebar.number_input("Initial Payment")
last_payment = st.sidebar.number_input("Last Payment")
credit_score = st.sidebar.number_input("Credit Score")
# house_number = st.sidebar.number_input("House Number")
sum_val = st.sidebar.number_input("Sum")

# Create a DataFrame from user input
user_data = pd.DataFrame({
    'Initial Payment': [initial_payment],
    'Last Payment': [last_payment],
    'Credit Score': [credit_score],
    # 'House Number': [house_number],
    'Sum': [sum_val]
})

# Button to predict loan repayment
if st.sidebar.button("Predict Loan Repayment"):
    # Check if the user input DataFrame has the same columns as the model expects
    if set(user_data.columns) == set(['Initial Payment', 'Last Payment', 'Credit Score', 'Sum']):
        # Make a prediction
        prediction = model.predict(user_data)
        
        # Display the prediction
        st.subheader("Prediction")
        if prediction[0] == 1:
            st.write("The customer is likely to return the loan.")
        else:
            st.write("The customer is unlikely to return the loan.")
    else:
        st.write("Please provide values for all input fields.")

  # Visualization of predicted results using a pie chart
    labels = ['Likely', 'Unlikely']
    sizes = [sum(prediction == 1), sum(prediction == 0)]
    
    st.subheader("Predicted Loan Repayment Distribution")
    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    st.pyplot(fig1)
