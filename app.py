import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
page_title="Amazon Product Rating Predictor",
page_icon="🛒",
layout="wide"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("best_model.pkl")

# ---------------- CUSTOM CSS ----------------

st.markdown("""

<style>

.stApp {
    background-color: #0f172a;
}

h1, h2, h3 {
    color: #60a5fa;
}

p, label, div {
    color: white;
    font-size: 17px;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

</style>

""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.title("ML Project")

st.sidebar.write("""
Amazon Product Rating Prediction System

Machine Learning Algorithms:

* Linear Regression
* Decision Tree
* Random Forest
  """)

# ---------------- TITLE ----------------

st.title("Amazon Product Rating Predictor")

# ---------------- IMAGE ----------------

st.image(
"https://images.unsplash.com/photo-1523475472560-d2df97ec485c",
use_container_width=True
)

# ---------------- PROJECT DESCRIPTION ----------------

st.markdown("""

## Project Overview

This Machine Learning project predicts Amazon product ratings based on:

* Product Price
* Product Title Length

### Best Model

Random Forest Regressor

""")

st.divider()

# ---------------- INPUTS ----------------

col1, col2 = st.columns(2)

with col1:

```
price = st.number_input(
    "Enter Product Price",
    min_value=0.0,
    value=500.0
)
```

with col2:

```
title_length = st.number_input(
    "Enter Title Length",
    min_value=1,
    value=10
)
```

# ---------------- PREDICTION ----------------

if st.button("Predict Rating"):

```
features = np.array([[price, title_length]])

prediction = model.predict(features)

st.success(f"Predicted Rating: {prediction[0]:.2f}")

# ---------------- METRICS ----------------

m1, m2, m3 = st.columns(3)

with m1:
    st.metric("Price", f"PKR {price}")

with m2:
    st.metric("Title Length", title_length)

with m3:
    st.metric("Predicted Rating", f"{prediction[0]:.2f}")

# ---------------- PROGRESS BAR ----------------

st.subheader("Prediction Confidence")

confidence = min(int((prediction[0] / 5) * 100), 100)

st.progress(confidence)

st.write(f"Confidence Score: {confidence}%")
```

# ---------------- SAMPLE GRAPH ----------------

st.divider()

st.subheader("Sample Rating Analysis")

chart_data = {
"Products": ["Phone", "Laptop", "Camera", "Speaker", "Tablet"],
"Ratings": [4.5, 4.7, 4.8, 4.0, 4.3]
}

st.bar_chart(
data={
"Ratings": chart_data["Ratings"]
}
)

# ---------------- FOOTER ----------------

st.divider()

st.markdown("""

### Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Railway Deployment

""")

st.caption("Machine Learning Project")
