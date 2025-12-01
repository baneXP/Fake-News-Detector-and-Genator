import streamlit as st
import sys, os

# Make sure Python can find detector/ and generator/
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from detector.detector_model import FakeNewsDetector
from generator.generator_model import FakeNewsGenerator

# Load detector model
detector = FakeNewsDetector("detector/fake_news_model.pkl")

# Load generator model
generator = FakeNewsGenerator()

st.set_page_config(page_title="Fake News Generator & Detector", layout="centered")

st.title("📰 Fake News Generator & Detector")
st.write("Generate fake news with AI and test if news is real or fake!")

# Tabs
tab1, tab2 = st.tabs(["🔍 Detect News", "📝 Generate Fake News"])

with tab1:
    st.subheader("Fake News Detector")
    user_input = st.text_area("Paste your news article/headline here:", height=150)
    if st.button("Detect"):
        if user_input.strip() == "":
            st.warning("Please enter some text to analyze.")
        else:
            prediction = detector.predict(user_input)
            # Convert 0/1 to FAKE/REAL
            result_text = "FAKE" if prediction == 1 else "REAL"
            if result_text == "FAKE":
                st.error("🚨 This looks like **FAKE NEWS**")
            else:
                st.success("✅ This looks like **REAL NEWS**")

with tab2:
    st.subheader("Fake News Generator")
    prompt = st.text_input("Enter a starting phrase:", "Breaking news: Scientists discovered")
    if st.button("Generate"):
        with st.spinner("Generating fake news..."):
            fake_news = generator.generate(prompt)
            st.write("📰 **Generated Fake News:**")
            st.info(fake_news)
