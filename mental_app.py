import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib

st.set_page_config(page_title="Scroll & Stress", page_icon="brain", layout="wide")
st.title("brain Scroll & Stress: Your Habits, Live")
st.markdown("**Drag. Click. Hover. See how small changes shift your risk.**")

# Load data + engineer features
@st.cache_data
def load_data():
    df = pd.read_csv("mental_health_social_media_dataset.csv")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    df["social_media_ratio"] = df["social_media_time_min"] / df["daily_screen_time_min"]
    df["neg_pos_ratio"] = (df["negative_interactions_count"] + 1) / (df["positive_interactions_count"] + 1)
    df["sleep_deficit"] = np.maximum(7 - df["sleep_hours"], 0)
    df["age_group"] = pd.cut(df["age"], bins=[0,18,25,35,50,100], 
                            labels=["Teen","Young Adult","Adult","Middle Age","Senior"])
    platform_risk = {"TikTok":4, "Instagram":3.5, "Twitter":3, "Snapchat":3, 
                     "YouTube":2.5, "Facebook":2, "WhatsApp":1, "Other":2.5}
    df["platform_risk"] = df["platform"].map(platform_risk).fillna(2.5)
    df["total_interactions"] = df["negative_interactions_count"] + df["positive_interactions_count"]
    return df

data = load_data()

# Load model & encoders
@st.cache_resource
def load_model():
    return (
        joblib.load('best_mental_health_model.pkl'),
        joblib.load('scaler.pkl'),
        joblib.load('label_encoder.pkl'),
        joblib.load('gender_encoder.pkl'),
        joblib.load('platform_encoder.pkl'),
        joblib.load('agegroup_encoder.pkl')
    )

model, scaler, le_mental, le_gender, le_platform, le_agegroup = load_model()

# Sidebar
st.sidebar.header("Explore")
view = st.sidebar.radio("View Mode", ["Explore Data", "Predict My Risk"])
platform_filter = st.sidebar.multiselect("Platforms", data["platform"].unique(), default=["TikTok","Instagram"])
age_filter = st.sidebar.slider("Age Range", 13, 69, (13, 40), step=1)
df = data[data["platform"].isin(platform_filter) & data["age"].between(age_filter[0], age_filter[1])]

# === INTERACTIVE DASHBOARD MODE ===
if view == "Explore Data":
    st.subheader("Live Data Explorer")

    # Row 1: Key Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Avg Screen Time", f"{df['daily_screen_time_min'].mean():.0f} min")
    col2.metric("Avg Sleep", f"{df['sleep_hours'].mean():.1f} h")
    col3.metric("Sleep Deficit", f"{df['sleep_deficit'].mean():.1f} h")
    col4.metric("% Stressed", f"{(df['mental_state']=='Stressed').mean()*100:.0f}%")

    # Row 2: 3D Scatter (THE STAR)
    st.subheader("3D Risk Space: Screen Time • Sleep • Stress")
    fig_3d = px.scatter_3d(
        df.sample(1000), 
        x="daily_screen_time_min", 
        y="sleep_hours", 
        z="stress_level",
        color="mental_state",
        size="social_media_ratio",
        hover_data=["age", "platform", "physical_activity_min"],
        color_discrete_map={"Healthy":"green", "At Risk":"orange", "Stressed":"red"},
        title="Drag to rotate • Hover for details"
    )
    fig_3d.update_layout(height=600)
    st.plotly_chart(fig_3d, use_container_width=True)

    # Row 3: Parallel Coordinates (see patterns across ALL features)
    st.subheader("Pattern Detector: Parallel Coordinates")
    plot_df = df[["age_group", "platform_risk", "social_media_ratio", "sleep_deficit", "physical_activity_min", "mental_state"]].copy()
    plot_df["age_group"] = plot_df["age_group"].astype(str)
    
    fig_par = px.parallel_coordinates(
        plot_df.sample(500),
        color="mental_state",
        labels={"platform_risk":"Platform Risk", "social_media_ratio":"Social %", "sleep_deficit":"Sleep Deficit", "physical_activity_min":"Exercise"},
        color_continuous_scale=["green","orange","red"],
        title="Trace lines from Healthy → Stressed"
    )
    st.plotly_chart(fig_par, use_container_width=True)

    # Row 4: Sunburst (hierarchy: Platform → Age → Mental State)
    st.subheader("Risk Hierarchy")
    sun_df = df.groupby(["platform", "age_group", "mental_state"]).size().reset_index(name="count")
    fig_sun = px.sunburst(sun_df, path=["platform", "age_group", "mental_state"], values="count",
                          color="mental_state", color_discrete_map={"Healthy":"lightgreen", "At Risk":"gold", "Stressed":"crimson"})
    st.plotly_chart(fig_sun, use_container_width=True)

# === PREDICTION MODE (now with LIVE WHAT-IF SLIDERS) ===
else:
    st.subheader("Your Risk, Right Now")
    
    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 13, 69, 25)
        gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        platform = st.selectbox("Main Platform", data["platform"].unique())
        screen = st.slider("Total Screen Time (min)", 60, 720, 300)
        social = st.slider("Social Media Time (min)", 0, screen, 180)
    
    with col2:
        neg = st.slider("Negative Interactions/day", 0, 15, 3)
        pos = st.slider("Positive Interactions/day", 0, 20, 8)
        sleep = st.slider("Sleep (hours)", 4.0, 11.0, 7.0, step=0.5)
        activity = st.slider("Exercise (min/day)", 0, 180, 30)
    
    # Auto-engineer features
    age_grp = pd.cut([age], bins=[0,18,25,35,50,100], labels=["Teen","Young Adult","Adult","Middle Age","Senior"])[0]
    input_df = pd.DataFrame({
        "age": [age],
        "gender": [le_gender.transform([gender])[0]],
        "platform": [le_platform.transform([platform])[0]],
        "age_group": [le_agegroup.transform([age_grp])[0]],
        "daily_screen_time_min": [screen],
        "social_media_time_min": [social],
        "social_media_ratio": [social/screen if screen > 0 else 0],
        "negative_interactions_count": [neg],
        "positive_interactions_count": [pos],
        "neg_pos_ratio": [(neg+1)/(pos+1)],
        "total_interactions": [neg+pos],
        "sleep_hours": [sleep],
        "sleep_deficit": [max(7-sleep, 0)],
        "physical_activity_min": [activity],
        "platform_risk": [data[data["platform"]==platform]["platform_risk"].iloc[0]]
    })

    pred_scaled = scaler.transform(input_df)
    prediction = model.predict(pred_scaled)[0]
    proba = model.predict_proba(pred_scaled)[0]
    state = le_mental.inverse_transform([prediction])[0]

    # Big result
    color = {"Healthy":"green", "At Risk":"orange", "Stressed":"red"}[state]
    st.markdown(f"<h1 style='color:{color}; text-align:center'>{state.upper()}</h1>", unsafe_allow_html=True)
    
    # Confidence bars
    fig = go.Figure()
    for i, s in enumerate(["Healthy", "At Risk", "Stressed"]):
        fig.add_trace(go.Bar(x=[s], y=[proba[i]], marker_color=["lightgreen","gold","crimson"][i]))
    fig.update_layout(title="Risk Confidence", yaxis_range=[0,1], height=300)
    st.plotly_chart(fig, use_container_width=True)

    # What-if suggestions
    st.info("Try adding 60min exercise or +1h sleep → watch the bars shift live")

st.caption("XGBoost • 93%+ accuracy • Not medical advice. Just awareness. Built with love.")