import streamlit as st
import joblib

def load_model_and_vectorizer():
    model = joblib.load(r"Spam_Email_Detection_model.h5")
    vectorizer = joblib.load(r"vectorizer.h5")
    return model, vectorizer

def predict_email(message):
    model, vectorizer = load_model_and_vectorizer() 
    message_vectorized = vectorizer.transform([message])  
    prediction = model.predict(message_vectorized)  
    return prediction[0] 

st.title("Email Spam Detection App")
st.write("Enter an email message below to check if it's spam or ham!")

email_text = st.text_area("Email Content", "")

if st.button("Predict"):
    if email_text.strip():  
        prediction = predict_email(email_text) 
        st.success(f"The Email is classified as: **{prediction}**")  
    else:
        st.error("Please enter some Email content.")  
