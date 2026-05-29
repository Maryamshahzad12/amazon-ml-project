import streamlit as st
import joblib
import numpy as np

# ---------------- PAGE SETTINGS ----------------

st.set_page_config(
page_title="Amazon Product Rating Predictor",
page_icon="🛒",
layout="centered"
)

# ---------------- LOAD MODEL ----------------

model = joblib.load("best_model.pkl")

# ---------------- CUSTOM CSS ----------------

st.markdown("""

<style>

.main {
    background-color: #0f172a;
}

h1, h2, h3, h4 {
    color: #60a5fa;
    text-align: center;
}

p, label, div {
    color: white;
}

.stButton>button {
    background-color: #2563eb;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    border: none;
}

.stButton>button:hover {
    background-color: #1d4ed8;
    color: white;
}

.stNumberInput input {
    background-color: #1e293b;
    color: white;
}

</style>

""", unsafe_allow_html=True)

# ---------------- TITLE ----------------

st.title("🛒 Amazon Product Rating Predictor")

st.image(
"https://images.unsplash.com/photo-1523475472560-d2df97ec485c",
use_container_width=True
)

# ---------------- PROJECT DESCRIPTION ----------------

st.markdown("""

## 📌 Project Overview

This Machine Learning project predicts Amazon product ratings using:

* 💰 Product Price
* 📝 Product Title Length

### 🤖 Machine Learning Models Used

* Linear Regression
* Decision Tree Regressor
* Random Forest Regressor

✅ Best Model Selected: **Random Forest Regressor**

""")

st.divider()

# ---------------- INPUT SECTION ----------------

st.subheader("📥 Enter Product Details")

price = st.number_input(
"💰 Enter Product Price",
min_value=0.0,
value=100.0
)

title_length = st.number_input(
"📝 Enter Product Title Length",
min_value=1,
value=10
)

# ---------------- PREDICTION BUTTON ----------------

if st.button("⭐ Predict Product Rating"):

```
features = np.array([[price, title_length]])

prediction = model.predict(features)

st.success(f"Predicted Product Rating: ⭐ {prediction[0]:.2f}")

if prediction[0] >= 4.5:

    st.balloons()

    st.markdown("""

    ## 😍 Excellent Product

    Customers are likely to love this product!

    """)

elif prediction[0] >= 4.0:

    st.markdown("""

    ## 👍 Good Product

    Product has good predicted ratings.

    """)

else:

    st.markdown("""

    ## 😐 Average Product

    Product may need improvement.

    """)
```

# ---------------- FOOTER ----------------

st.divider()

st.markdown("""

## 👨‍💻 Technologies Used

* Python
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Railway Deployment

---

## 📊 Features Used

* Product Price
* Product Title Length

---

## 🚀 Deployment Platform

Railway Cloud Platform

""")

st.caption("Made with ❤️ using Machine Learning")
