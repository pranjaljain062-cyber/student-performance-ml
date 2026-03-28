import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="📊",
    layout="centered"
)


# Title
st.title("📊 Student Performance Predictor")

st.markdown("""
This app predicts student performance based on study habits and lifestyle factors.

Fill the details below and click **Predict**.
""")

# Inputs
col1, col2 = st.columns(2)

with col1:
    study = st.number_input("Study Hours", 0, 12)
    attendance = st.number_input("Attendance (%)", 0, 100)
    sleep = st.number_input("Sleep Hours", 0, 12)

with col2:
    internet = st.number_input("Internet Usage", 0, 10)
    job = st.selectbox("Part Time Job", ["Yes", "No"])
    previous = st.number_input("Previous Score", 0, 100)

# Convert job
job = 1 if job == "Yes" else 0

# Load data
df = pd.read_csv("file.csv")
df["PartTimeJob"] = df["PartTimeJob"].map({"Yes":1,"No":0})

# Feature Engineering
df["Efficiency"] = df["PreviousScore"] / df["StudyHours"]
df["Lifestyle"] = df["SleepHours"] - df["InternetUsage"]
df["Engagement"] = df["StudyHours"] + df["Attendance"]/10

X = df[[
    "StudyHours","Attendance","SleepHours","InternetUsage",
    "PartTimeJob","Efficiency","Lifestyle","Engagement"
]]
y = df["Performance"]

# Train model
model = RandomForestClassifier(n_estimators=100)
model.fit(X, y)

# Button
predict_btn = st.button("Predict Performance")
if predict_btn:

    efficiency = previous / study if study != 0 else 0
    lifestyle = sleep - internet
    engagement = study + (attendance / 10)

    input_data = pd.DataFrame([[
        study, attendance, sleep, internet,
        1 if job == "Yes" else 0,
        efficiency, lifestyle, engagement
    ]], columns=X.columns)

    result = model.predict(input_data)

    if result[0] == "High":
        st.success("🔥 High Performance")
    elif result[0] == "Medium":
        st.warning("⚠️ Medium Performance")
    else:
        st.error("❌ Low Performance")

