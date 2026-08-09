import streamlit as st
import osmnx as ox
import networkx as nx
import folium
from streamlit_folium import st_folium
import joblib
import pandas as pd
import random

# ── PAGE CONFIG ──────────────────────────────────
st.set_page_config(
    page_title="Flood Evacuation Route Optimizer",
    page_icon="🌊",
    layout="wide"
)

st.title("🌊 Flood Evacuation Route Optimizer")
st.markdown("Find the **safest evacuation route** during floods using ML + GIS")

# ── LOAD MODEL ───────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("model_small.pkl")

model = load_model()

# ── SIDEBAR ──────────────────────────────────────
st.sidebar.header("⚙️ Flood Conditions")

monsoon  = st.sidebar.slider("Monsoon Intensity",   1, 10, 5)
drainage = st.sidebar.slider("Drainage Quality",    1, 10, 5)
urbanize = st.sidebar.slider("Urbanization Level",  1, 10, 5)
deforest = st.sidebar.slider("Deforestation",       1, 10, 3)
infra    = st.sidebar.slider("Infrastructure Decay",1, 10, 4)

st.sidebar.markdown("---")
st.sidebar.markdown("**Legend**")
st.sidebar.markdown("🔵 Blue = Normal route")
st.sidebar.markdown("🟢 Green = Safe route")
st.sidebar.markdown("🔴 Red roads = High flood risk")
st.sidebar.markdown("🟠 Orange = Medium risk")

# ── PREDICT FLOOD RISK ───────────────────────────
def predict_risk(noise=0):
    sample = pd.DataFrame([{
        "MonsoonIntensity":                monsoon + noise,
        "TopographyDrainage":              drainage,
        "RiverManagement":                 random.randint(2, 7),
        "Deforestation":                   deforest,
        "Urbanization":                    urbanize,
        "ClimateChange":                   random.randint(3, 8),
        "DamsQuality":                     random.randint(2, 8),
        "Siltation":                       random.randint(2, 7),
        "AgriculturalPractices":           random.randint(2, 7),
        "Encroachments":                   random.randint(2, 8),
        "IneffectiveDisasterPreparedness": random.randint(2, 8),
        "DrainageSystems":                 drainage,
        "CoastalVulnerability":            random.randint(2, 7),
        "Landslides":                      random.randint(1, 6),
        "Watersheds":                      random.randint(2, 7),
        "DeterioratingInfrastructure":     infra,
        "PopulationScore":                 urbanize,
        "WetlandLoss":                     random.randint(2, 8),
        "InadequatePlanning":              random.randint(2, 8),
        "PoliticalFactors":                random.randint(2, 7),
    }]).clip(1, 10)
    return model.predict(sample)[0]

# ── LOAD ROAD MAP ────────────────────────────────
@st.cache_resource
def load_graph():
    return ox.graph_from_place("Indiranagar, Bangalore, India",
                                network_type="drive")

# ── MAIN BUTTON ──────────────────────────────────
if st.button("🔍 Find Safe Evacuation Route", use_container_width=True):

    with st.spinner("Loading road network..."):
        G = load_graph()

    with st.spinner("Calculating flood risk on all roads..."):
        for u, v, data in G.edges(data=True):
            noise = random.randint(-2, 2)
            risk  = predict_risk(noise)
            data["flood_risk"]  = risk
            data["safe_weight"] = data.get("length", 100) * (1 + risk * 5)

    # Pick start and end
    nodes      = list(G.nodes())
    start_node = nodes[10]
    end_node   = nodes[200]

    # Find routes
    try:
        normal_route = nx.shortest_path(G, start_node, end_node, weight="length")
        safe_route   = nx.shortest_path(G, start_node, end_node, weight="safe_weight")
    except nx.NetworkXNoPath:
        st.error("No path found. Please try again.")
        st.stop()

    # ── BUILD FOLIUM MAP ─────────────────────────
    center = ox.geocode("Indiranagar, Bangalore, India")
    m = folium.Map(location=center, zoom_start=15)

    # Color roads by flood risk
    for u, v, data in G.edges(data=True):
        risk = data.get("flood_risk", 0.5)
        color = "#FF4444" if risk > 0.55 else "#FFA500" if risk > 0.45 else "#44BB44"
        try:
            y1, x1 = G.nodes[u]["y"], G.nodes[u]["x"]
            y2, x2 = G.nodes[v]["y"], G.nodes[v]["x"]
            folium.PolyLine([[y1,x1],[y2,x2]],
                            color=color, weight=2, opacity=0.6).add_to(m)
        except:
            pass

    # Draw normal route (blue)
    normal_coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in normal_route]
    folium.PolyLine(normal_coords, color="blue",
                    weight=6, opacity=0.8,
                    tooltip="Normal Shortest Route").add_to(m)

    # Draw safe route (green)
    safe_coords = [[G.nodes[n]["y"], G.nodes[n]["x"]] for n in safe_route]
    folium.PolyLine(safe_coords, color="green",
                    weight=6, opacity=0.9,
                    tooltip="Safe Evacuation Route").add_to(m)

    # Markers
    folium.Marker(normal_coords[0],
                  tooltip="START",
                  icon=folium.Icon(color="green", icon="play")).add_to(m)
    folium.Marker(normal_coords[-1],
                  tooltip="SAFE ZONE",
                  icon=folium.Icon(color="red", icon="flag")).add_to(m)

    # ── DISPLAY ──────────────────────────────────
    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("🗺️ Evacuation Map")
        st_folium(m, width=750, height=500)

    with col2:
        st.subheader("📊 Risk Summary")
        overall = predict_risk()
        if overall > 0.55:
            st.error(f"🔴 HIGH RISK\n\n{overall:.2%} flood probability")
        elif overall > 0.45:
            st.warning(f"🟡 MEDIUM RISK\n\n{overall:.2%} flood probability")
        else:
            st.success(f"🟢 LOW RISK\n\n{overall:.2%} flood probability")

        st.markdown("---")
        st.metric("Normal route steps", len(normal_route))
        st.metric("Safe route steps",   len(safe_route))

else:
    st.info("👈 Adjust flood conditions on the left sidebar, then click the button!")
    m = folium.Map(location=[12.9784, 77.6408], zoom_start=14)
    st_folium(m, width=750, height=400)