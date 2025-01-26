# -*- coding: utf-8 -*-
# Streamlit deployment code
import streamlit as st
import pickle
import time
headind_css="""
<style>
[data-testid="stHeading"]{
color: #000000;
text-align: center;
opacity: 1.0;
}
"""
out_cs="""
<style>
[data-testid="stAlert"]{
background-color: #A94A4A;
color: #FFF6DA;
opacity: 1.0;
}
</style>
"""
head_css="""
<style>
[data-testid="stHeader"]{
background-image: url("https://unsplash.com/photos/FNFBN4-GdlQ/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8MjB8fHdlYnNpdGUlMjBiYWNrZ3JvdW5kfGVufDB8fHx8MTczNzgzOTk0MXww&force=true&w=2400");
opacity: 1.0;
}
</style>
"""
page_pg_img="""
<style>
[data-testid="stAppViewContainer"]{
position: Relative;
background-image: url("https://unsplash.com/photos/FNFBN4-GdlQ/download?ixid=M3wxMjA3fDB8MXxzZWFyY2h8MjB8fHdlYnNpdGUlMjBiYWNrZ3JvdW5kfGVufDB8fHx8MTczNzgzOTk0MXww&force=true&w=2400");
opacity: 1.0;
width: 100%;
height: 100%;
background-size: auto auto;
}
"""
def main():
    st.title("News Detection")
    with open('random_forest_model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)

    with open('tfidf_vectorizer.pkl', 'rb') as vectorizer_file:
        vectorizer = pickle.load(vectorizer_file)

    # User input
    user_input = st.text_area(" ",placeholder="enter the news to analyze")
    l,ll,lm,m,rm,lr,r = st.columns(7)
    if m.button("Analyze"):
        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
                my_bar.progress(percent_complete + 1, text=progress_text)
                time.sleep(0.1)
                my_bar.empty()
        
    if user_input.strip():
    # Transform user input
        input_vectorized = vectorizer.transform([user_input])
        # Predict
        prediction = model.predict(input_vectorized)[0]
        # Display result
        if prediction == 1:
            st.success("This news is Real.")
        else:
            st.error("This news is Fake.")
    else:
        st.warning("Please enter some text.")
    st.markdown(head_css,unsafe_allow_html=True)
    st.markdown(out_cs,unsafe_allow_html=True)
    st.markdown(page_pg_img,unsafe_allow_html=True)
    st.markdown(headind_css,unsafe_allow_html=True)
    st.markdown(
        """
        <style>
        .stProgress > div > div > div > div {
        background-color: green;
        }
        </style>""",
        unsafe_allow_html=True,)        

if __name__ == "__main__":
    main()
