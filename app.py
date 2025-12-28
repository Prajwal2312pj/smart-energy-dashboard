import streamlit as st
import numpy as np
import joblib
import plotly.express as px

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Smart Energy Dashboard",
    page_icon="⚡",
    layout="centered"
)

# ---------------- THEME TOGGLE ---------------- #
theme = st.sidebar.radio("🎨 Theme Mode", ["Dark", "Light"])

if theme == "Dark":
    bg_color = "#0e1117"
    card_color = "#161b22"
    text_color = "white"
    accent = "#58a6ff"
else:
    bg_color = "#f5f7fa"
    card_color = "#ffffff"
    text_color = "#111111"
    accent = "#2563eb"

# ---------------- CUSTOM CSS ---------------- #
st.markdown(f"""
<style>
body {{
    background-color: {bg_color};
    color: {text_color};
}}

.metric-box {{
    background-color: {card_color};
    padding: 20px;
    border-radius: 15px;
    text-align: center;
    box-shadow: 0px 0px 20px rgba(0,0,0,0.2);
}}

.metric-title {{
    font-size: 15px;
    color: gray;
}}

.metric-value {{
    font-size: 30px;
    font-weight: bold;
    color: {accent};
}}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("energy_prediction_model.pkl")
features = joblib.load("model_features.pkl")

# ---------------- HEADER ---------------- #
st.markdown("## ⚡ Smart Energy Consumption Dashboard")
st.markdown(
    "AI-powered **monthly energy & bill prediction** with interactive visualization"
)
st.markdown("---")

# ---------------- SIDEBAR INPUTS ---------------- #
st.sidebar.markdown("## 🔧 Input Parameters")

hour = st.sidebar.slider("🕒 Hour of Day", 0, 23, 12)
day = st.sidebar.slider("📅 Day", 1, 31, 15)
month = st.sidebar.selectbox("🗓 Month", list(range(1, 13)), index=4)
voltage = st.sidebar.slider("⚡ Voltage (V)", 210.0, 250.0, 230.0)
current = st.sidebar.slider("🔌 Current (A)", 0.0, 10.0, 5.0)
unit_price = st.sidebar.number_input("💰 Unit Price (₹/kWh)", 1.0, 15.0, 6.0)

# ---------------- ML PREDICTION ---------------- #
input_data = np.array([[hour, day, month, voltage, current]])
predicted_power = model.predict(input_data)[0]  # kW

monthly_energy = predicted_power * 24 * 30 / 60
monthly_bill = monthly_energy * unit_price

# ---------------- METRIC CARDS ---------------- #
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Predicted Power</div>
        <div class="metric-value">{predicted_power:.3f} kW</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Monthly Energy</div>
        <div class="metric-value">{monthly_energy:.2f} kWh</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-title">Estimated Bill</div>
        <div class="metric-value">₹ {monthly_bill:.2f}</div>
    </div>
    """, unsafe_allow_html=True)

# ---------------- PLOTLY ANIMATED CHART ---------------- #
st.markdown("## 📈 Interactive Daily Energy Trend")

days = np.arange(1, 31)
daily_energy = (predicted_power * 24 / 60) * np.ones(30)

chart_data = {
    "Day": days,
    "Energy (kWh)": daily_energy
}

fig = px.line(
    chart_data,
    x="Day",
    y="Energy (kWh)",
    markers=True,
    title="Daily Energy Consumption Pattern",
    template="plotly_dark" if theme == "Dark" else "plotly_white"
)

fig.update_traces(line=dict(width=4))
fig.update_layout(
    title_font_size=20,
    xaxis_title="Day of Month",
    yaxis_title="Energy (kWh)",
)

st.plotly_chart(fig, use_container_width=True)

# ---------------- INSIGHTS ---------------- #
st.markdown("## 🧠 Smart Insights")

if monthly_energy > 200:
    st.warning("⚠ High energy consumption detected. Consider reducing heavy appliance usage.")
elif monthly_energy > 100:
    st.info("ℹ Moderate energy usage. Optimization can save money.")
else:
    st.success("✅ Energy usage is efficient. Great job!")

# ---------------- FOOTER ---------------- #
st.markdown("---")
st.markdown(
    "🚀 Built by **Prajwal Dattatray Jadhav** | Smart Energy ML Project"
)
