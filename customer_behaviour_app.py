
#imports
import numpy as np
import streamlit as st
import pickle

#load model
def load_model():
    try:
        with open("Customer_Behaviour.pkl", "rb") as file:
            model = pickle.load(file)

        scaler = None
        try:
            with open("scaler.pkl", "rb") as file:
                scaler = pickle.load(file)
        except:
            st.warning("Scaler not found or invalid")
            return model, scaler

    except FileNotFoundError:
        st.error("Model file not found")
        return None, None

    return model, scaler

model,scaler = load_model()

def genderInput(gender_input):
    if gender_input == "Male":
            return 0
    else:
        return 1

#convert predictions
def resultOutput(result):
    if result ==1:
        return "Yes"
    else:
        return "No"
def customer_satisfaction_prediction(gender_input,age_input,salary_input):
    try:
        gender_value = genderInput(gender_input)
        age_value = float(age_input)
        salary_value = float(salary_input)

        input_data = np.array([[gender_value,age_value,salary_value]])

        if scaler is None or not hasattr(scaler,'transform'):
            return"Error: Scaler not available or invalid"
        scaled_data = scaler.transform(input_data)

        prediction = model.predict(scaled_data)
        probabilities = model.predict_proba(scaled_data)
        predicted_purchase = int(prediction[0])

        confidence = probabilities[0][predicted_purchase] 

        return predicted_purchase, confidence 
    except Exception as e:
        return f"prediction Error:{e}",None
    
    #main
def main():
    st.title("Customer_ Behaviour prediction App")
    gender_input = st.selectbox('Select_Gender',['Male','Female'])
    age_input = st.number_input("Enter Age",min_value = 18,max_value=100,value=30)
    salary_input= st.number_input("Enter Estimated Salary",min_value=0,value=50000,step=500)

    if st.button("Predict Customer Purchase"):
        if model is None:
            st.error("Model not loaded properly.Please check the file")

        result,confidence = customer_satisfaction_prediction(gender_input,age_input,salary_input)

        if isinstance(result,str)and (result.startswith("Error")or result.startswith("Prediction Error")):
            st.error(result)
        else:
            result_output = resultOutput(result)
            st.success(f"Will customer Purchase?: {result_output}")
            if confidence is not None:
                st.info(f"Confidence: {confidence:.2%}")
if __name__ =="__main__":
    main()