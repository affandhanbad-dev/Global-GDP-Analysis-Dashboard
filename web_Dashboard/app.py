import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pickle
from helper import HelperClass
import sklearn

# -------------------- Page Setup --------------------
st.set_page_config(
    page_title="🌍 Global GDP Analysis Dashboard",
    page_icon="💹",
    layout="wide"
)

# -------------------- Custom Background Style --------------------
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to bottom right, #eef2f3, #8e9eab);
    background-attachment: fixed;
}
[data-testid="stHeader"] {
    background: rgba(255, 255, 255, 0);
}
[data-testid="stSidebar"] {
    background-color: #F5F5F5;
}
[data-testid="stSidebar"] * {
    color: black !important;
}
h1, h2, h3, h4, h5 {
    color: #008000;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# -------------------- Load Model --------------------
model = pickle.load(open('Global_GDP_EDA.sav', 'rb'))

# -------------------- App Header --------------------
st.markdown("<h1 style='text-align: center;'>🌍 Global GDP Analysis & Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: #444;'>Analyze, visualize, and predict GDP performance across regions.</p>",
    unsafe_allow_html=True)
st.markdown("---")

# -------------------- Prediction Section --------------------
with st.container():
    st.header("🔮 GDP Prediction System")
    features = st.text_input("💡 Enter Input Features (comma-separated)", placeholder="Example: 0.7, 0.85, 0.65, 1.2")
    if st.button("🚀 Predict GDP"):
        try:
            features = np.array([features.split(',')], dtype=float)
            gdp_pred = model.predict(features).reshape(1, -1)
            st.success(f"💰 Predicted GDP Per Capita: **${gdp_pred[0]:,.2f}**")
        except Exception as e:
            st.error(f"⚠️ Invalid input format: {e}")

st.markdown("---")

# -------------------- Sidebar --------------------
st.sidebar.title("📂 Data Uploader & Controls")
file = st.sidebar.file_uploader("Upload CSV File", type=['csv'])

if file is not None:
    df = pd.read_csv(file)
    st.sidebar.success("✅ File uploaded successfully!")
    st.sidebar.markdown("---")

    region, countries, countries_counts = HelperClass.basic_counts(df)
    st.sidebar.metric("🌎 Total Regions", region)
    st.sidebar.metric("🏳️ Total Countries", countries)
    st.sidebar.write("📊 **Countries per Region:**")
    st.sidebar.write(countries_counts)

    show_dash = st.sidebar.button("📈 Show Analysis Dashboard")

else:
    st.sidebar.info("⬆️ Upload a CSV file to begin analysis.")
    show_dash = False

# -------------------- Main Dashboard --------------------
if file is not None and show_dash:
    df = HelperClass.ConvertToFloatAndFillMissValues(df)

    st.markdown("## 🧭 Data Overview")
    st.dataframe(df.head(10), use_container_width=True)
    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📉 Average GDP, Literacy & Agriculture by Region")
        result = HelperClass.AverageRegionsGDPLiteracyAgriculture(df)
        st.dataframe(result, use_container_width=True)

    with col2:
        st.subheader("📦 Data Aggregation by Region")
        data_agg = HelperClass.DataAgg(df)
        st.dataframe(data_agg, use_container_width=True)

    st.markdown("---")

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("💰 Top 20 Countries by GDP Per Capita")
        HelperClass.top20_gdp_bar(df)
    with col4:
        st.subheader("📘 Literacy Distribution by Region")
        HelperClass.literacy_pie(df)

    st.markdown("---")

    col5, col6 = st.columns(2)
    with col5:
        st.subheader("🩺 Infant Mortality Rate by Region")
        HelperClass.infant_mortality_pie(df)
    with col6:
        st.subheader("👥 Population Distribution by Region")
        HelperClass.population_pie(df)

    st.markdown("---")

    st.subheader("📊 Correlation Heatmap of Numerical Features")
    HelperClass.correlation_heatmap(df)

    st.markdown("---")

    col7, col8 = st.columns(2)
    with col7:
        st.subheader("🌏 Top 5 Asian Countries (GDP & Literacy)")
        HelperClass.AsiaFiveRegionGDP(df)
    with col8:
        st.subheader("🌍 Top 5 Countries per Region by GDP")
        HelperClass.EachReginGDP(df)

# -------------------- Footer --------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:gray;'>📊 Built using Streamlit | Developed by <b>Affan</b></p>",
    unsafe_allow_html=True
)
