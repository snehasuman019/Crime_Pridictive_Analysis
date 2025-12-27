import streamlit as st
import pandas as pd
import numpy as np
import os

from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium

# --------------------------------------------------
# PAGE CONFIG (MUST BE FIRST)
# --------------------------------------------------
st.set_page_config(
    page_title="Crime Predictive Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --------------------------------------------------
# DARK MODE + BACKGROUND IMAGE
# --------------------------------------------------
st.markdown(
    """
    <style>
    .stApp {
        background: 
            linear-gradient(rgba(14,17,23,0.85), rgba(14,17,23,0.85)),
            url("image.webp");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        color: #e6edf3;
    }

    section[data-testid="stSidebar"] {
        background-color: rgba(22,27,34,0.95);
    }

    section[data-testid="stSidebar"] * {
        color: #e6edf3 !important;
    }

    h1, h2, h3, h4 {
        color: #e6edf3;
    }

    p, span, label {
        color: #c9d1d9;
    }

    div[data-testid="metric-container"] {
        background-color: rgba(22,27,34,0.9);
        border: 1px solid #30363d;
        padding: 12px;
        border-radius: 10px;
    }

    .stDataFrame {
        background-color: rgba(14,17,23,0.9);
    }

    .stButton > button {
        background-color: #238636;
        color: white;
        border-radius: 6px;
        border: none;
    }

    .stButton > button:hover {
        background-color: #2ea043;
    }

    hr {
        border-color: #30363d;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.markdown(
    """
    <div style="text-align:center;">
        <h1>📊 Crime Predictive Analytics Dashboard</h1>
        <p style="font-size:18px;">
        Interactive Machine Learning Dashboard with Maps, Predictions & Insights
        </p>
    </div>
    <hr>
    """,
    unsafe_allow_html=True
)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------
base_dir = os.path.dirname(__file__)
data_path = os.path.join(base_dir, "data", "crime_preprocessed.csv")

df = pd.read_csv(data_path)
df.columns = df.columns.str.strip()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------
st.sidebar.markdown("## 🧭 Navigation")

section = st.sidebar.radio(
    "",
    [
        "📁 Data Overview",
        "📈 Regression",
        "🧠 Classification",
        "🧩 Clustering",
        "🔻 PCA",
        "⚙️ Model Performance",
        "🗺️ Crime Map"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    """
    **Project:** Crime Predictive Analytics  
    **Tech:** Python, ML, Streamlit  
    **Theme:** Dark Mode + Geospatial Analytics
    """
)

# --------------------------------------------------
# DATA OVERVIEW
# --------------------------------------------------
if section == "📁 Data Overview":
    st.subheader("📁 Dataset Preview")
    st.dataframe(df.head(50))

    st.subheader("📊 Basic Statistics")
    st.write(df.describe())

    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Dataset",
        csv,
        "crime_analysis_report.csv",
        "text/csv"
    )

# --------------------------------------------------
# REGRESSION
# --------------------------------------------------
elif section == "📈 Regression":
    st.subheader("📈 Crime Hour Prediction")

    year = int(df.Year.min())

    col1, col2 = st.columns(2)
    with col1:
        month = st.slider("Month", 1, 12, 6)
        lat = st.slider("Latitude", float(df.LATITUDE.min()), float(df.LATITUDE.max()), float(df.LATITUDE.mean()))
    with col2:
        lon = st.slider("Longitude", float(df.LONGITUDE.min()), float(df.LONGITUDE.max()), float(df.LONGITUDE.mean()))

    X = df[['Year', 'Month', 'LATITUDE', 'LONGITUDE']]
    y = df['Hour']

    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LinearRegression().fit(X_train, y_train)

    pred = model.predict(pd.DataFrame([[year, month, lat, lon]], columns=X.columns))[0]
    st.success(f"🕒 Predicted Crime Hour: **{pred:.2f}**")

# --------------------------------------------------
# CLASSIFICATION
# --------------------------------------------------
elif section == "🧠 Classification":
    st.subheader("🧠 Crime Shift Prediction")

    col1, col2 = st.columns(2)
    with col1:
        year = st.selectbox("Year", sorted(df.Year.unique()))
        month = st.selectbox("Month", list(range(1, 13)))
        offense = st.selectbox("Offense", sorted(df.OFFENSE.unique()))
    with col2:
        lat = st.slider("Latitude", float(df.LATITUDE.min()), float(df.LATITUDE.max()), float(df.LATITUDE.mean()))
        lon = st.slider("Longitude", float(df.LONGITUDE.min()), float(df.LONGITUDE.max()), float(df.LONGITUDE.mean()))

    X = df[['Year', 'Month', 'LATITUDE', 'LONGITUDE', 'OFFENSE']]
    y = df['SHIFT']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    clf = DecisionTreeClassifier(random_state=42).fit(X_train, y_train)

    pred = clf.predict(pd.DataFrame([[year, month, lat, lon, offense]], columns=X.columns))[0]
    acc = accuracy_score(y_test, clf.predict(X_test))

    st.success(f"🕐 Predicted Shift: **{pred}**")
    st.metric("Accuracy", round(acc, 3))

# --------------------------------------------------
# CLUSTERING
# --------------------------------------------------
elif section == "🧩 Clustering":
    st.subheader("🧩 Crime Hotspot Detection")

    k = st.slider("Clusters (K)", 2, 8, 4)
    df['Cluster'] = KMeans(n_clusters=k, random_state=42).fit_predict(df[['LATITUDE', 'LONGITUDE']])

    st.scatter_chart(df, x="LONGITUDE", y="LATITUDE", color="Cluster")

# --------------------------------------------------
# PCA
# --------------------------------------------------
elif section == "🔻 PCA":
    st.subheader("🔻 Principal Component Analysis")

    num_df = df.select_dtypes(include=[np.number]).copy()
    num_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    num_df.dropna(axis=1, how="all", inplace=True)

    X_scaled = StandardScaler().fit_transform(SimpleImputer(strategy="mean").fit_transform(num_df))
    X_scaled = np.nan_to_num(X_scaled)

    pca_df = pd.DataFrame(PCA(n_components=2).fit_transform(X_scaled), columns=["PC1", "PC2"])
    st.scatter_chart(pca_df, x="PC1", y="PC2")

# --------------------------------------------------
# MODEL PERFORMANCE
# --------------------------------------------------
elif section == "⚙️ Model Performance":
    st.subheader("⚙️ Model Performance")

    X = df[['Year', 'Month', 'LATITUDE', 'LONGITUDE', 'OFFENSE']]
    y = df['SHIFT']

    scores = cross_val_score(LogisticRegression(max_iter=1000), X, y, cv=5)
    rf = RandomForestClassifier(n_estimators=100, random_state=42).fit(X, y)

    c1, c2, c3 = st.columns(3)
    c1.metric("CV Accuracy", round(scores.mean(), 3))
    c2.metric("Features", len(X.columns))
    c3.metric("Models", 5)

    imp = pd.DataFrame({"Feature": X.columns, "Importance": rf.feature_importances_})
    st.bar_chart(imp.set_index("Feature"))

# --------------------------------------------------
# CRIME MAP
# --------------------------------------------------
elif section == "🗺️ Crime Map":
    st.subheader("🗺️ Crime Location Map")

    m = folium.Map(location=[df.LATITUDE.mean(), df.LONGITUDE.mean()], zoom_start=11)
    cluster = MarkerCluster().add_to(m)

    for _, row in df.sample(min(500, len(df))).iterrows():
        folium.Marker(
            [row['LATITUDE'], row['LONGITUDE']],
            popup=f"Offense: {row['OFFENSE']} | Shift: {row['SHIFT']}"
        ).add_to(cluster)

    st_folium(m, width=1200, height=600)

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.markdown(
    """
    <hr>
    <div style="text-align:center; font-size:14px;">
    🚀 Built with Streamlit, Machine Learning & Geospatial Analytics  
    <br>
    Crime Predictive Analytics Project
    </div>
    """,
    unsafe_allow_html=True
)
