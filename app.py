import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

st.set_page_config(page_title="Routerisk - Accident Death Forecast", layout="centered")
st.title("🚦 Routerisk: Predict Accidental Deaths in India")
st.markdown("Enter accident statistics to estimate **traffic accident deaths** and get a **risk rating** for your region.")

# Input Fields
road_cases = st.number_input("🚗 Road Accidents - Cases", min_value=0)
road_injured = st.number_input("🚑 Road Accidents - Injured", min_value=0)
road_died = st.number_input("💀 Road Accidents - Died", min_value=0)
rail_cases = st.number_input("🚆 Railway Accidents - Cases", min_value=0)
rail_injured = st.number_input("🚑 Railway Accidents - Injured", min_value=0)
rail_died = st.number_input("💀 Railway Accidents - Died", min_value=0)
crossing_cases = st.number_input("⚠️ Railway Crossing Accidents - Cases", min_value=0)
crossing_injured = st.number_input("🚑 Railway Crossing Accidents - Injured", min_value=0)
crossing_died = st.number_input("💀 Railway Crossing Accidents - Died", min_value=0)
traffic_cases = st.number_input("📊 Total Traffic Accidents - Cases", min_value=0)
traffic_injured = st.number_input("🤕 Total Traffic Accidents - Injured", min_value=0)

@st.cache_resource
def get_model():
    df = pd.read_csv("data/ADSI_Table_1A.2.csv")
    df = df.drop(columns=['Sl. No.'])
    df.columns = [
        'State', 'Road_Accidents_Cases', 'Road_Accidents_Injured', 'Road_Accidents_Died',
        'Railway_Accidents_Cases', 'Railway_Accidents_Injured', 'Railway_Accidents_Died',
        'Railway_Crossing_Cases', 'Railway_Crossing_Injured', 'Railway_Crossing_Died',
        'Total_Traffic_Cases', 'Total_Traffic_Injured', 'Total_Traffic_Died'
    ]
    df = df.dropna()
    X = df.drop(columns=['State', 'Total_Traffic_Died'])
    y = df['Total_Traffic_Died']
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)
    return model

model = get_model()

if st.button("🔮 Predict"):
    input_data = np.array([[
        road_cases, road_injured, road_died,
        rail_cases, rail_injured, rail_died,
        crossing_cases, crossing_injured, crossing_died,
        traffic_cases, traffic_injured
    ]])

    prediction = int(model.predict(input_data)[0])
    st.success(f"🧮 **Predicted Deaths**: {prediction}")

    if prediction < 3000:
        st.markdown("### 🟢 **Low Risk**")
    elif prediction < 7000:
        st.markdown("### 🟡 **Medium Risk**")
    else:
        st.markdown("### 🔴 **High Risk**")
