import streamlit as st
import joblib

# Load trained model and vectorizer
model = joblib.load("spam_classifier.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Set page configuration
st.set_page_config(page_title="Spam Classifier", page_icon="🚀", layout="centered")

# Custom styling
st.markdown(
    """
    <style>
    /* Increase font size for the title */
    .big-title {
        font-size: 60px;  /* Increase size */
        font-weight: bold;
        text-align: center;
        color: #ffffff;
        padding: 20px;  /* Add space around */
        background-color: #222222; /* Dark background */
        border-radius: 10px; /* Smooth edges */
        display: inline-block;
    }
 .title-container {
        text-align: center;
    }

    /* Fix input text visibility */
    .stTextArea textarea {
        background-color: #ffffff !important;
        color: #000000 !important;
        font-size: 16px;
        border-radius: 8px;
    }

    /* Style the button */
    .stButton button {
        background-color: #FF4B4B;
        color: white;
        font-size: 18px;
        border-radius: 10px;
        padding: 10px;
    }

    /* Style for the result display */
    .result-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        font-size: 20px;
        font-weight: bold;
    }

    /* Remove Streamlit's default sidebar toggle */
    [data-testid="collapsedControl"] {
        display: none;
    }
<div class="title-container">
        <p class="big-title">🚀 Spam Message Classifier</p>
    </style>
    """,
    unsafe_allow_html=True
)

# Main title
st.markdown('<p class="big-title">🚀 Spam Message Classifier</p>', unsafe_allow_html=True)
st.write("Enter a message below to check if it's **Spam** or **Ham**.")

# User input
user_input = st.text_area(
    "Enter your message here...",
    placeholder="E.g., Congratulations! You've won a free iPhone! Click here to claim.",
)

# Classification button
if st.button("🚀 Classify Message"):
    if user_input.strip():
        # Transform input text
        input_vectorized = vectorizer.transform([user_input])

        # Predict
        prediction = model.predict(input_vectorized)[0]

        # Display result
        if prediction == 1:
            st.markdown(
                '<div class="result-box" style="background-color: #ffcccc; color: red;">'
                '🚨 This message is <span style="color:red;">Spam!</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="result-box" style="background-color: #d4edda; color: green;">'
                '✅ This message is <span style="color:green;">Ham (Not Spam)</span>'
                '</div>',
                unsafe_allow_html=True,
            )
    else:
        st.warning("⚠️ Please enter a message to classify.")
