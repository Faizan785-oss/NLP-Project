
import os
import pandas as pd
import streamlit as st
import joblib

st.set_page_config(
    page_title="NLP Sentiment Analysis",
    page_icon="💬",
    layout="wide"
)

st.title("NLP Sentiment Analysis")
st.write("Analyze customer reviews using machine learning.")

# Load the trained model
if not os.path.exists("sentiment_model.joblib"):
    st.error("Model not found. Complete WEEK 2 training first.")
    st.stop()

model = joblib.load("sentiment_model.joblib")

# Create application tabs
tab1, tab2 = st.tabs([
    "Sentiment Prediction",
    "Dataset Analysis"
])

#  SENTIMENT PREDICTION
with tab1:
    st.subheader("Enter Customer Review")

    review = st.text_area(
        "Customer review",
        placeholder="The product is excellent..."
    )

    if st.button("Analyze Sentiment"):
        if not review.strip():
            st.warning("Please enter a review.")
        else:
            prediction = model.predict([review.strip()])[0]

            if prediction == "Positive":
                st.success("Sentiment: Positive")
            elif prediction == "Negative":
                st.error("Sentiment: Negative")
            else:
                st.info("Sentiment: Neutral")

#  DATASET ANALYSIS
with tab2:
    st.subheader("Customer Review Analysis")

    if os.path.exists("cleaned_reviews.csv"):
        df = pd.read_csv("cleaned_reviews.csv")

        st.metric("Total Reviews", len(df))

        st.write("Sentiment Distribution")
        st.bar_chart(df["sentiment"].value_counts())

        st.write("Sample Reviews")
        st.dataframe(
            df[["rating", "review_text", "sentiment"]].head(20),
            use_container_width=True
        )
    else:
        st.warning("Run to create the cleaned dataset.")
