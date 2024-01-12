import pickle
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("vectorizer_HSD.pkl", "rb")))
loaded_model = pickle.load(open('trained_HSD.sav', 'rb'))
def hate_speech_detection():
    import streamlit as st 
    st.title("Hate Speech Detection")
    user = st.text_area("Enter any Tweet: ")
    if len(user) < 1:
        st.write(" ")
    else:
        sample = user
        data = loaded_vec.transform(np.array([sample]))
        a = loaded_model.predict(data) 
        st.title(a) 
hate_speech_detection()
