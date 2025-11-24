import streamlit as st
from scorer import score_transcript

st.title("Communication Skills Scoring Tool")

text = st.text_area("Paste transcript here")

duration = st.number_input("Enter duration in seconds", min_value=1, max_value=600)

if st.button("Score"):
    results = score_transcript(text, duration)
    st.write(results)
