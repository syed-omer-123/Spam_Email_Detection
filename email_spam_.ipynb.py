#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib


# In[2]:


df = pd.read_csv(r"C:\Users\SYED OMER HUSSAINI\OneDrive\Desktop\TECHZONE\Python\spam.csv", encoding="ISO-8859-1")


# In[3]:


X = df["Message"]  
y = df["Category"]  


# In[4]:


cv = CountVectorizer() 
X_vector = cv.fit_transform(X) 


# In[5]:


nb_model = MultinomialNB()
nb_model.fit(X_vector, y)


# In[6]:


joblib.dump(nb_model, r"C:\Users\SYED OMER HUSSAINI\OneDrive\Desktop\TECHZONE\Trained Models\Spam_Email_Detection\Spam_Email_Detection_model.h5")
joblib.dump(cv, r"C:\Users\SYED OMER HUSSAINI\OneDrive\Desktop\TECHZONE\Trained Models\Spam_Email_Detection\vectorizer.h5")


# In[7]:


def load_model_and_vectorizer(model_path, vectorizer_path):
    model = joblib.load(r"C:\Users\SYED OMER HUSSAINI\OneDrive\Desktop\TECHZONE\Trained Models\Spam_Email_Detection\Spam_Email_Detection_model.h5")
    vectorizer = joblib.load(r"C:\Users\SYED OMER HUSSAINI\OneDrive\Desktop\TECHZONE\Trained Models\Spam_Email_Detection\vectorizer.h5")
    return model, vectorizer


# In[8]:


def predict(statement):
    model, vectorizer = load_model_and_vectorizer(r"C:\Users\SYED OMER HUSSAINI\OneDrive\Desktop\TECHZONE\Trained Models\Spam_Email_Detection\spam_detection_model.h5",r"C:\Users\SYED OMER HUSSAINI\OneDrive\Desktop\TECHZONE\Trained Models\Spam_Email_Detectionvectorizer.h5")
    statement_vectorized = vectorizer.transform(statement)
    prediction = model.predict(statement_vectorized)
    return prediction


# In[9]:


test_statement = input("Enter the email you received: ")
result = predict([test_statement])
print(f"Prediction: {result[0]}")


# In[ ]:




