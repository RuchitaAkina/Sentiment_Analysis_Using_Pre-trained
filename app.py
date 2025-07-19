
import os
os.environ["USE_TF"] = "0"  # Disable TensorFlow

import streamlit as st
from transformers import pipeline

st.set_page_config(page_title="YouTube Comment Sentiment Analyzer")

st.title("YouTube Comment Sentiment Analyzer")
st.markdown("Enter **one or more** YouTube comments below (one per line):")

comments_input = st.text_area("Enter comments")

if st.button("Analyze"):
    if comments_input.strip() == "":
        st.warning("Please enter at least one comment.")
    else:
        classifier = pipeline("sentiment-analysis")
        comments = comments_input.strip().split('\n')
        
        for comment in comments:
            if comment.strip() == "":
                continue
            result = classifier(comment)[0]
            st.markdown(f"**Comment**: {comment}")
            st.markdown(f"- **Sentiment**: {result['label']}")
            st.markdown(f"- **Confidence**: {round(result['score'], 2)}")
            st.markdown("---")
