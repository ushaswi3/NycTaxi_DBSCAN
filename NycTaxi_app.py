import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score

# ---------------------------------------------------
# PAGE CONFIG
# ---------------------------------------------------
st.set_page_config(
    page_title="Taxi Pickup Hotspot Detection",
    page_icon="🚕",
    layout="wide"
)

# ---------------------------------------------------
# LOAD DATA
# ---------------------------------------------------
@st.cache_data
def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    file_path = os.path.join(
        base_dir,
        "NewYorkCityTaxiTripDuration.csv"
    )
    return pd.read_csv(file_path)

df = load_data()

# ---------------------------------------------------
# TITLE
# ---------------------------------------------------
st.title("🚕 NYC Taxi Pickup Hotspot Detection using DBSCAN")

st.markdown("""
This application identifies natural taxi pickup hotspots using **DBSCAN Clustering**.

### Experiments
- eps = 0.2
- eps = 0.3
- eps = 0.5

The app evaluates:
- Number of Clusters
- Number of Noise Points
- Noise Ratio
- Silhouette Score
""")

# ---------------------------------------------------
# DISPLAY DATA
# ---------------------------------------------------
st.subheader("1️⃣ First 5 Rows of Dataset")
st.dataframe(df.head())

# ---------------------------------------------------
# FEATURE SELECTION
# ---------------------------------------------------
required_cols = [
    "pickup_latitude",
    "pickup_longitude"
]

if all(col in df.columns for col in required_cols):

    X = df[required_cols].copy()

    st.subheader("2️⃣ Selected Features")
    st.dataframe(X.head())

    # ---------------------------------------------------
    # DATA PREPROCESSING
    # ---------------------------------------------------
    X = X.dropna()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    st.subheader("3️⃣ Data Scaling Completed")

    # ---------------------------------------------------
    # DBSCAN EXPERIMENTS
    # ---------------------------------------------------
    eps_values = [0.2, 0.3, 0.5]

    results = []

    st.subheader("4️⃣ - 8️⃣ DBSCAN Experiments & Evaluation")

    for eps in eps_values:

        dbscan = DBSCAN(
            eps=eps,
            min_samples=5
        )

        labels = dbscan.fit_predict(X_scaled)

        n_clusters = len(set(labels)) - (
            1 if -1 in labels else 0
        )

        noise_points = np.sum(labels == -1)

        noise_ratio = noise_points / len(labels)

        # ---------------------------------------------------
        # SILHOUETTE SCORE
        # ---------------------------------------------------
        mask = labels != -1

        if np.sum(mask) > 1:

            unique_clusters = len(set(labels[mask]))

            if unique_clusters > 1:
                silhouette = silhouette_score(
                    X_scaled[mask],
                    labels[mask]
                )
            else:
                silhouette = None

        else:
            silhouette = None

        results.append({
            "eps": eps,
            "labels": labels,
            "clusters": n_clusters,
            "noise_points": noise_points,
            "noise_ratio": noise_ratio,
            "silhouette": silhouette
        })

        st.markdown(f"### Experiment (eps = {eps})")

        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Clusters",
            n_clusters
        )

        col2.metric(
            "Noise Points",
            noise_points
        )

        col3.metric(
            "Noise Ratio",
            f"{noise_ratio:.2%}"
        )

        col4.metric(
            "Silhouette Score",
            (
                f"{silhouette:.4f}"
                if silhouette is not None
                else "Not Applicable"
            )
        )

    # ---------------------------------------------------
    # VISUALIZATION
    # ---------------------------------------------------
    st.subheader("9️⃣ Cluster Visualizations")

    for result in results:

        labels = result["labels"]

        fig, ax = plt.subplots(figsize=(8, 6))

        scatter = ax.scatter(
            X["pickup_longitude"],
            X["pickup_latitude"],
            c=labels,
            cmap="tab20",
            s=10
        )

        noise_mask = labels == -1

        ax.scatter(
            X.loc[noise_mask, "pickup_longitude"],
            X.loc[noise_mask, "pickup_latitude"],
            c="red",
            s=15,
            label="Noise Points"
        )

        ax.set_title(
            f"DBSCAN Clustering (eps = {result['eps']})"
        )

        ax.set_xlabel("Pickup Longitude")
        ax.set_ylabel("Pickup Latitude")

        ax.legend()

        st.pyplot(fig)

    # ---------------------------------------------------
    # BEST MODEL SELECTION
    # ---------------------------------------------------
    st.subheader("🔟 Best Model Selection")

    valid_results = [
        r for r in results
        if r["silhouette"] is not None
    ]

    if len(valid_results) > 0:

        best_model = max(
            valid_results,
            key=lambda x: x["silhouette"]
        )

        st.success(
            f"Best eps value = {best_model['eps']}"
        )

        st.markdown("### Best Model Statistics")

        st.write(
            f"**Number of Clusters:** {best_model['clusters']}"
        )

        st.write(
            f"**Noise Points:** {best_model['noise_points']}"
        )

        st.write(
            f"**Noise Ratio:** {best_model['noise_ratio']:.2%}"
        )

        st.write(
            f"**Silhouette Score:** {best_model['silhouette']:.4f}"
        )

    else:

        st.warning(
            "Silhouette Score Not Applicable for any experiment."
        )

else:

    st.error(
        "Dataset must contain the columns "
        "'pickup_latitude' and 'pickup_longitude'."
    )