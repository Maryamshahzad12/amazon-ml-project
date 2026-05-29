import streamlit as st
import joblib
import numpy as np
import pandas as pd

# PAGE CONFIG

st.set_page_config(
    page_title="Amazon Product Rating Predictor",
    page_icon="🛒",
    layout="wide"
)

# LOAD MODEL

model = joblib.load("best_model.pkl")

# CUSTOM CSS

st.markdown("""
<style>

.stApp {
    background-color: #081229;
}

h1, h2, h3 {
    color: #4da6ff;
    font-weight: bold;
}

p, label, div {
    color: white;
    font-size: 17px;
}

[data-testid="stSidebar"] {
    background-color: #111827;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 3.2em;
    width: 100%;
    font-size: 18px;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    transform: scale(1.02);
}

.metric-card {
    background-color: #111827;
    padding: 20px;
    border-radius: 15px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)

# SIDEBAR

st.sidebar.title("ML Project Dashboard")

st.sidebar.write("""
This project predicts Amazon product ratings using Machine Learning models.

### Algorithms Used
- Linear Regression
- Decision Tree
- Random Forest

### Best Model
Random Forest Regressor
""")

# TITLE

st.title("Amazon Product Rating Predictor")

st.write("Predict product ratings using Machine Learning.")

# TOP IMAGE

st.image(
    "https://images.unsplash.com/photo-1519389950473-47ba0277781c",
    use_container_width=True
)

st.divider()

# PROJECT OVERVIEW

st.subheader("Project Overview")

st.write("""
This Machine Learning project predicts Amazon product ratings based on:

- Product Price
- Product Title Length

The system uses trained ML models to estimate product ratings.
""")

st.divider()

# INPUT SECTION

st.subheader("Enter Product Details")

col1, col2 = st.columns(2)

with col1:
    price = st.number_input(
        "Product Price",
        min_value=0.0,
        value=500.0
    )

with col2:
    title_length = st.number_input(
        "Title Length",
        min_value=1,
        value=10
    )

# PREDICTION

if st.button("Predict Rating"):

    features = np.array([[price, title_length]])

    prediction = model.predict(features)

    predicted_rating = round(prediction[0], 2)

    st.success(f"Predicted Rating: {predicted_rating}")

    st.divider()

    # METRICS

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric("Price", f"PKR {price}")

    with m2:
        st.metric("Title Length", title_length)

    with m3:
        st.metric("Predicted Rating", predicted_rating)

    # PROGRESS BAR

    st.subheader("Prediction Score")

    confidence = min(int((predicted_rating / 5) * 100), 100)

    st.progress(confidence)

    st.write(f"Prediction Confidence: {confidence}%")

st.divider()

# SAMPLE GRAPH

st.subheader("Sample Product Ratings")

chart_data = pd.DataFrame({
    "Products": ["Phone", "Laptop", "Camera", "Tablet", "Speaker"],
    "Ratings": [4.5, 4.7, 4.8, 4.3, 4.0]
})

st.bar_chart(chart_data.set_index("Products"))

st.divider()

# SECOND IMAGE

st.image(
    "https://images.unsplash.com/photo-1498050108023-c5249f4df085",
    use_container_width=True
)

# FOOTER

st.divider()

st.subheader("Technologies Used")

st.write("""
- Python
- Pandas
- NumPy
- Scikit-Learn
- Streamlit
- Railway Deployment
""")

st.caption("Machine Learning Project")
