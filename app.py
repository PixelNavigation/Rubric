import streamlit as st
from scorer import score_transcript

st.set_page_config(page_title="Communication Scoring Tool", layout="wide")

st.title("🗣️ Communication Skills Scoring Tool")

st.markdown("Paste transcript text below:")

text = st.text_area("Transcript", height=200)

duration = st.number_input("Audio duration (seconds)", min_value=1)

if st.button("Score"):
    if len(text.strip()) < 5:
        st.error("Please enter valid transcript text.")
    else:
        results = score_transcript(text, duration)

        st.success(f"Overall Score: {results['overall_score']:.2f} / 100")

        st.subheader("Detailed Scores")
        for r in results["details"]:
            st.markdown(f"""
            ### {r['criterion']}
            **Score:** {r['score']}  
            **Feedback:** {r['feedback']}
            ---
            """)