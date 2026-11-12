"""Streamlit demo: run with  `streamlit run app.py`  (needs insurance_model.pkl from the notebook)."""
import joblib, pandas as pd, streamlit as st

bundle = joblib.load("insurance_model.pkl")
model, cols = bundle["model"], bundle["columns"]

st.title("Medical Insurance Cost Estimator")
age = st.slider("Age", 18, 64, 35)
sex = st.radio("Sex", ["male", "female"], horizontal=True)
bmi = st.number_input("BMI", 15.0, 55.0, 28.0, 0.1)
children = st.slider("Children", 0, 5, 0)
smoker = st.radio("Smoker", ["no", "yes"], horizontal=True)
region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

row = {"age": age, "sex": int(sex == "male"), "bmi": bmi, "children": children,
       "smoker": int(smoker == "yes"),
       "region_northwest": int(region == "northwest"),
       "region_southeast": int(region == "southeast"),
       "region_southwest": int(region == "southwest")}
pred = float(model.predict(pd.DataFrame([row])[cols])[0])
st.metric("Estimated yearly charges (USD)", f"{pred:,.0f}")
