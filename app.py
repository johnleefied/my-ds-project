import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

st.set_page_config(page_title="Rental Valuation & Market Intelligence", layout="wide")

@st.cache_resource
def load_pipeline():
    return joblib.load("models/price_model.joblib")

@st.cache_data
def load_data():
    df = pd.read_csv("data/raw/listings.csv")
    df = df.dropna(subset=["price", "latitude", "longitude", "neighbourhood_group", "room_type"])
    return df[(df["price"] >= 20) & (df["price"] <= 800)]

pipeline = load_pipeline()
df = load_data()

st.title("🏡 Rental Valuation & Market Intelligence Dashboard")
st.markdown("Estimate market benchmarks for rental listings and inspect neighborhood price distributions.")

# --- SIDEBAR INPUTS ---
st.sidebar.header("Listing Parameters")

neighbourhoods = sorted(df["neighbourhood_group"].unique().tolist())
room_types = sorted(df["room_type"].unique().tolist())

selected_nhood = st.sidebar.selectbox("Borough / Neighbourhood Group", neighbourhoods)
selected_room = st.sidebar.selectbox("Room Type", room_types)
min_nights = st.sidebar.slider("Minimum Nights Stay", min_value=1, max_value=30, value=3)
reviews = st.sidebar.slider("Number of Historical Reviews", min_value=0, max_value=300, value=25)

# Calculate borough center coordinates
borough_coords = df[df["neighbourhood_group"] == selected_nhood][["latitude", "longitude"]].mean()

input_record = pd.DataFrame([{
    "latitude": borough_coords["latitude"],
    "longitude": borough_coords["longitude"],
    "minimum_nights": min_nights,
    "number_of_reviews": reviews,
    "neighbourhood_group": selected_nhood,
    "room_type": selected_room
}])

# --- INFERENCE ---
predicted_price = pipeline.predict(input_record)[0]

# --- MAIN DASHBOARD LAYOUT ---
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Valuation Benchmark")
    st.metric(
        label="Estimated Nightly Rate",
        value=f"${predicted_price:,.2f}"
    )
    st.write("---")
    st.markdown("**Active Listing Inputs:**")
    st.dataframe(input_record.T.rename(columns={0: "Input Value"}), use_container_width=True)

with col2:
    st.subheader(f"Geographic Market Spread: {selected_nhood}")
    nhood_sample = df[df["neighbourhood_group"] == selected_nhood].sample(
        min(400, len(df[df["neighbourhood_group"] == selected_nhood]))
    )
    
    # Modern Plotly 6+ compatibility
    map_func = getattr(px, "scatter_map", getattr(px, "scatter_mapbox", None))
    style_key = "map_style" if hasattr(px, "scatter_map") else "mapbox_style"

    fig = map_func(
        nhood_sample,
        lat="latitude",
        lon="longitude",
        color="price",
        size="price",
        hover_name="name",
        color_continuous_scale="Viridis",
        zoom=10.5,
        center={"lat": float(borough_coords["latitude"]), "lon": float(borough_coords["longitude"])},
        title=f"Sample Listings in {selected_nhood}",
        **{style_key: "open-street-map"}
    )
    fig.update_layout(margin={"r": 0, "t": 35, "l": 0, "b": 0})
    st.plotly_chart(fig, use_container_width=True)

# Distribution Chart
st.write("---")
st.subheader("Price Distribution by Room Type Across All Boroughs")
fig_box = px.box(
    df.sample(2000, random_state=42),
    x="neighbourhood_group",
    y="price",
    color="room_type",
    points=False,
    labels={"neighbourhood_group": "Borough", "price": "Nightly Rate ($)", "room_type": "Room Type"}
)
st.plotly_chart(fig_box, use_container_width=True)
