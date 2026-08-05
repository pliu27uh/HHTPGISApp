import os
import re
import json
import numpy as np
import pandas as pd
import requests
import geopandas as gpd
import folium
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from scipy.spatial import cKDTree
from streamlit_folium import st_folium


# ==========================================================
# PAGE CONFIG
# ==========================================================

st.set_page_config(
    page_title="National Hydrogen Transportation Fuel Supply (alpha)",
    layout="wide",
    initial_sidebar_state="collapsed"  # Collapses the sidebar on initial page load
)


# ==========================================================
# GLOBAL STYLE
# ==========================================================

st.markdown(
    """
<style>

html, body, [class*="css"] {
    font-size: 14pt;
}

/* Reduce Streamlit main title font size */
h1 {
    font-size: 16pt !important; /* Adjust font size value as needed */
}

table {
    border-collapse: collapse !important;
}

td, th {
    border: none !important;
    font-size: 14pt !important;
    text-align: center !important;
}

</style>
""",
    unsafe_allow_html=True,
)


# ==========================================================
# US STATE DICTIONARY & CENTROIDS FOR SPATIAL LOOKUP
# ==========================================================

STATE_NAME_TO_ABBR = {
    "ALABAMA": "AL", "ALASKA": "AK", "ARIZONA": "AZ", "ARKANSAS": "AR", "CALIFORNIA": "CA",
    "COLORADO": "CO", "CONNECTICUT": "CT", "DELAWARE": "DE", "FLORIDA": "FL", "GEORGIA": "GA",
    "HAWAII": "HI", "IDAHO": "ID", "ILLINOIS": "IL", "INDIANA": "IN", "IOWA": "IA",
    "KANSAS": "KS", "KENTUCKY": "KY", "LOUISIANA": "LA", "MAINE": "ME", "MARYLAND": "MD",
    "MASSACHUSETTS": "MA", "MICHIGAN": "MI", "MINNESOTA": "MN", "MISSISSIPPI": "MS", "MISSOURI": "MO",
    "MONTANA": "MT", "NEBRASKA": "NE", "NEVADA": "NV", "NEW HAMPSHIRE": "NH", "NEW JERSEY": "NJ",
    "NEW MEXICO": "NM", "NEW YORK": "NY", "NORTH CAROLINA": "NC", "NORTH DAKOTA": "ND", "OHIO": "OH",
    "OKLAHOMA": "OK", "OREGON": "OR", "PENNSYLVANIA": "PA", "RHODE ISLAND": "RI", "SOUTH CAROLINA": "SC",
    "SOUTH DAKOTA": "SD", "TENNESSEE": "TN", "TEXAS": "TX", "UTAH": "UT", "VERMONT": "VT",
    "VIRGINIA": "VA", "WASHINGTON": "WA", "WEST VIRGINIA": "WV", "WISCONSIN": "WI", "WYOMING": "WY",
    "DISTRICT OF COLUMBIA": "DC"
}

US_STATE_CENTROIDS = {
    'AL': (32.806671, -86.791130), 'AK': (61.370716, -152.404419), 'AZ': (33.729759, -111.431221),
    'AR': (34.969704, -92.373123), 'CA': (36.116203, -119.681564), 'CO': (39.059811, -105.311104),
    'CT': (41.597782, -72.755371), 'DE': (39.318523, -75.507141), 'FL': (27.766279, -81.686783),
    'GA': (33.040619, -83.643074), 'HI': (21.094318, -157.498337), 'ID': (44.240459, -114.478828),
    'IL': (40.349457, -88.986137), 'IN': (39.849426, -86.258278), 'IA': (42.011539, -92.920231),
    'KS': (38.526600, -96.726486), 'KY': (37.668140, -84.670067), 'LA': (31.169546, -91.867805),
    'ME': (44.693947, -69.381927), 'MD': (39.063946, -76.802101), 'MA': (42.230171, -71.530106),
    'MI': (43.326618, -84.536095), 'MN': (45.694454, -93.900192), 'MS': (32.741646, -89.678696),
    'MO': (38.456085, -92.288368), 'MT': (46.921925, -110.454353), 'NE': (41.125370, -98.268082),
    'NV': (38.313515, -117.055374), 'NH': (43.452492, -71.563896), 'NJ': (40.298960, -74.521011),
    'NM': (34.840515, -106.248482), 'NY': (42.165726, -74.948051), 'NC': (35.630066, -79.806419),
    'ND': (47.528912, -99.784012), 'OH': (40.388783, -82.764915), 'OK': (35.565342, -96.928917),
    'OR': (44.572021, -122.070938), 'PA': (40.590752, -77.209755), 'RI': (41.680893, -71.511780),
    'SC': (33.856892, -80.945007), 'SD': (44.299782, -99.438828), 'TN': (35.747845, -86.692345),
    'TX': (31.054487, -97.563461), 'UT': (40.150032, -111.862434), 'VT': (44.045876, -72.710686),
    'VA': (37.769337, -78.169968), 'WA': (47.400902, -121.490494), 'WV': (38.491226, -80.954453),
    'WI': (44.268543, -89.616508), 'WY': (42.755966, -107.302490), 'DC': (38.897438, -77.026817)
}

STATE_CENTROID_ABBRS = list(US_STATE_CENTROIDS.keys())
STATE_CENTROID_COORDS = [US_STATE_CENTROIDS[s] for s in STATE_CENTROID_ABBRS]
state_centroid_tree = cKDTree(STATE_CENTROID_COORDS)


def normalize_state_str(st_input):
    """
    Normalizes state input strings (e.g. 'Texas', 'Texas Gulf', 'CA', 'California')
    to a two-letter postal code abbreviation.
    """
    if not st_input or pd.isna(st_input):
        return None
    st_clean = str(st_input).strip().upper()
    if st_clean in STATE_NAME_TO_ABBR:
        return STATE_NAME_TO_ABBR[st_clean]
    if len(st_clean) == 2 and st_clean in STATE_NAME_TO_ABBR.values():
        return st_clean
    for full_name, abbr in STATE_NAME_TO_ABBR.items():
        if full_name in st_clean:
            return abbr
    for word in st_clean.split():
        if len(word) == 2 and word in STATE_NAME_TO_ABBR.values():
            return word
    return st_clean


# ==========================================================
# GLOBAL UTILITY FUNCTIONS
# ==========================================================


def find_column(columns, target_keywords):
    """
    Finds a column name matching search terms, ignoring case, underscores, and spacing.
    """
    if isinstance(target_keywords, str):
        target_keywords = [target_keywords]

    # Precise Case-Insensitive Match
    for col in columns:
        col_clean = str(col).strip().lower().replace("_", " ")
        for kw in target_keywords:
            kw_clean = str(kw).strip().lower().replace("_", " ")
            if col_clean == kw_clean:
                return col

    # Substring Match
    for col in columns:
        col_clean = str(col).strip().lower().replace("_", " ")
        for kw in target_keywords:
            kw_clean = str(kw).strip().lower().replace("_", " ")
            if kw_clean in col_clean:
                return col
    return None


def fmt_money(v):
    """
    Globally available currency formatter for tables and tooltips.
    """
    if pd.isna(v) or v == 0.0:
        return "N/A"
    return f"${v:.3f}" if abs(v) < 0.1 else f"${v:.2f}"


# ==========================================================
# ARC GIS PIPELINE & SALINE AQUIFER SERVICES
# ==========================================================

PIPELINE_SERVICE_URL = "https://geo.dot.gov/server/rest/services/Hosted/Natural_Gas_Pipelines_US_EIA/FeatureServer/0/query"
SALINE_SERVICE_URL = "https://arcgis.netl.doe.gov/server/rest/services/EDXSpatialCaches/NATCARB_Saline_Poly_v1502_27_2025/MapServer/4/query"

PIPELINE_SHP_PATH = "pipeline_layer.shp"
SALINE_SHP_PATH = "saline_aquifers_layer.shp"


@st.cache_data(ttl=86400)
def load_pipeline_layer():
    # Load directly from local shapefile if present
    if os.path.exists(PIPELINE_SHP_PATH):
        try:
            gdf = gpd.read_file(PIPELINE_SHP_PATH)
            return json.loads(gdf.to_json())
        except Exception:
            pass

    if "YOUR_ARCGIS" in PIPELINE_SERVICE_URL:
        return None

    try:
        metadata = requests.get(
            PIPELINE_SERVICE_URL, params={"f": "json"}, timeout=60
        ).json()

        max_records = metadata.get("maxRecordCount", 1000)
        features = []
        offset = 0

        while True:
            params = {
                "where": "1=1",
                "outFields": "*",
                "returnGeometry": "true",
                "f": "geojson",
                "resultOffset": offset,
                "resultRecordCount": max_records,
            }

            response = requests.get(
                PIPELINE_SERVICE_URL, params=params, timeout=60
            )

            data = response.json()
            batch = data.get("features", [])

            if not batch:
                break

            features.extend(batch)
            offset += max_records

            if len(batch) < max_records:
                break

        geojson_data = {"type": "FeatureCollection", "features": features}

        # Save to local shapefile for future fast-loading
        if features:
            try:
                gdf = gpd.GeoDataFrame.from_features(features)
                gdf.crs = "EPSG:4326"
                gdf.to_file(PIPELINE_SHP_PATH)
            except Exception:
                pass

        return geojson_data
    except Exception:
        return None


@st.cache_data(ttl=86400)
def load_saline_aquifers_layer():
    """
    Loads NATCARB Saline Aquifer Polygons from local shapefile if available,
    otherwise fetches from DOE NETL ArcGIS REST Service and exports to shapefile.
    """
    # Load directly from local shapefile if present
    if os.path.exists(SALINE_SHP_PATH):
        try:
            gdf = gpd.read_file(SALINE_SHP_PATH)
            return json.loads(gdf.to_json())
        except Exception:
            pass

    try:
        params = {
            "where": "objectid >= 0 AND objectid < 1999",
            "outFields": "*",
            "returnGeometry": "true",
            "f": "geojson",
            "outSR": "4326",
        }
        response = requests.get(SALINE_SERVICE_URL, params=params, timeout=60)
        if response.status_code == 200:
            data = response.json()
            features = data.get("features", [])

            # Save to local shapefile for future fast-loading
            if features:
                try:
                    gdf = gpd.GeoDataFrame.from_features(features)
                    gdf.crs = "EPSG:4326"
                    gdf.to_file(SALINE_SHP_PATH)
                except Exception:
                    pass

            return data
    except Exception:
        pass
    return None


def pipeline_style(feature):
    props = feature.get("properties", {}) or {}

    pipeline_type = ""
    for k, v in props.items():
        key_clean = str(k).lower()
        if "type" in key_clean or "cat" in key_clean or "class" in key_clean:
            if v is not None:
                pipeline_type = str(v).lower()
                break

    if not pipeline_type:
        pipeline_type = " ".join(str(v).lower() for v in props.values() if v)

    if "interstate" in pipeline_type:
        return {"color": "#0066FF", "weight": 3, "opacity": 0.25}
    if "intrastate" in pipeline_type:
        return {"color": "#FF2222", "weight": 3, "opacity": 0.25}

    return {"color": "#707070", "weight": 2, "opacity": 0.6}


def saline_style(feature):
    """
    Blue-green filled polygons with 30% fill opacity for NATCARB Saline Aquifers.
    """
    return {
        "fillColor": "#379E9B",  # Blue-Green / Teal
        "color": "#005F56",      # Stroke color
        "weight": 1,
        "fillOpacity": 0.10,     # 30% fill opacity
        "opacity": 0.50,
    }


# ==========================================================
# HYDROGEN LOCATIONS & DATA INGESTION PIPELINE
# ==========================================================


@st.cache_data
def load_projects():
    try:
        df = pd.read_excel("LTCHDataSet.xlsx")
        return df
    except FileNotFoundError:
        return pd.DataFrame({
            "Site": [
                "Texas Gulf",
                "California Central",
                "Arizona Solar",
                "Colorado Wind",
                "Utah Hub",
                "Nevada Desert",
                "New Mexico Basin",
                "Oklahoma Plains",
            ],
            "State": ["TX", "CA", "AZ", "CO", "UT", "NV", "NM", "OK"],
            "Latitude": [31.0, 36.8, 34.5, 39.1, 40.5, 37.5, 35.2, 35.4],
            "Longitude": [
                -99.0,
                -119.5,
                -111.8,
                -105.5,
                -111.9,
                -116.5,
                -106.6,
                -97.5,
            ],
            "Electricity Price ($/kWh)": [
                0.045,
                0.065,
                0.038,
                0.042,
                0.040,
                0.055,
                0.035,
                0.039,
            ],
            "Water Price ($/gallon)": [
                0.005,
                0.008,
                0.004,
                0.005,
                0.004,
                0.007,
                0.003,
                0.005,
            ],
            "Natural Gas Price ($/MSCF)": [
                4.2,
                5.5,
                4.8,
                4.5,
                4.0,
                5.0,
                3.8,
                4.1,
            ],
            "CO₂ Storage Costs ($/kg H2)": [
                0.02,
                0.03,
                0.02,
                0.02,
                0.01,
                0.03,
                0.01,
                0.02,
            ],
            "CO₂ Transport Costs ($/kg H₂)": [
                0.01,
                0.02,
                0.01,
                0.01,
                0.01,
                0.02,
                0.01,
                0.01,
            ],
            "Gasoline Price ($/gal)": [
                3.20,
                3.80,
                3.40,
                3.30,
                3.10,
                3.60,
                3.00,
                3.25,
            ],
            "Diesel Price ($/gal)": [
                3.90,
                4.30,
                4.10,
                4.00,
                3.80,
                4.20,
                3.70,
                3.90,
            ],
            "CNG Price ($/GGE)": [2.50, 3.10, 2.70, 2.60, 2.40, 2.90, 2.30, 2.50],
            "BEV CO₂ Emissions (lb CO₂/kWh)": [
                0.85,
                0.92,
                0.75,
                0.80,
                0.78,
                0.95,
                0.88,
                0.82,
            ],
            "ICEV CO2 Emissions per mile (kg/mile)": [
                1.45,
                1.50,
                1.40,
                1.42,
                1.38,
                1.52,
                1.48,
                1.41,
            ],
            "Levelized Cost of Transport (LCT) ($/kg H₂)": [
                0.45,
                0.60,
                0.50,
                0.48,
                0.40,
                0.55,
                0.38,
                0.42,
            ],
            "Levelized Cost of Refueling Stations (LCRS) ($/kg H₂)": [
                1.20,
                1.50,
                1.30,
                1.25,
                1.15,
                1.40,
                1.10,
                1.18,
            ],
            "Formation Name and Identifier": [
                "Frio Fm",
                "Monterey Shale",
                "Luke Basin",
                "Denver Basin",
                "Paradox Basin",
                "Nevada Play",
                "San Juan",
                "Anadarko",
            ],
            "Lithology": [
                "Sandstone",
                "Shale",
                "Salt Dome",
                "Sandstone",
                "Limestone",
                "Volcanic Tuff",
                "Sandstone",
                "Shale",
            ],
            "Saline Aquifer Location": [
                "TX Gulf Coast",
                "CA Central Valley",
                "AZ Deep Saline",
                "CO Front Range",
                "UT Paradox",
                "NV Basin Fill",
                "NM San Juan",
                "OK Anadarko",
            ],
            "Pipeline Operator": [
                "Enbridge H2",
                "Kinder Morgan",
                "Energy Transfer",
                "TC Energy",
                "Williams Cos",
                "Oneok Inc",
                "Enterprise Products",
                "Plains All American",
            ],
            "Pipe Type": [
                "Transmission",
                "Distribution",
                "Transmission",
                "Transmission",
                "Gathering",
                "Transmission",
                "Gathering",
                "Transmission",
            ],
            "Pipeline distances (mi)": [
                14.2,
                38.5,
                8.1,
                22.4,
                11.7,
                44.0,
                6.3,
                18.9,
            ],
        })


@st.cache_data
def load_tornado_data():
    try:
        df = pd.read_csv("TornadoPlotDataFile.csv")
        return df
    except FileNotFoundError:
        states = ["TX", "CA", "AZ", "CO", "UT", "NV", "NM", "OK"]
        pathways = ["SMR", "SMRCC", "Electrolysis"]
        vars_map = {
            "SMR": [
                ("Natural Gas Price", "$/MSCF", 2.50, 4.20, 6.50, 1.80, 2.80),
                ("Water Price", "$/gal", 0.002, 0.005, 0.009, 2.30, 2.50),
                ("Electricity Price", "$/kWh", 0.030, 0.045, 0.075, 2.20, 2.65),
            ],
            "SMRCC": [
                ("Natural Gas Price", "$/MSCF", 2.50, 4.20, 6.50, 2.10, 3.10),
                ("CO2 Storage Costs", "$/kg H2", 0.010, 0.020, 0.040, 2.40, 2.70),
                ("Electricity Price", "$/kWh", 0.030, 0.045, 0.075, 2.45, 2.85),
                ("Water Price", "$/gal", 0.002, 0.005, 0.009, 2.50, 2.68),
            ],
            "Electrolysis": [
                ("Electricity Price", "$/kWh", 0.025, 0.045, 0.080, 3.10, 5.20),
                ("Water Price", "$/gal", 0.002, 0.005, 0.009, 4.10, 4.40),
                ("CapEx Multiplier", "Ratio", 0.80, 1.00, 1.30, 3.80, 4.80),
            ],
        }

        records = []
        for st_code in states:
            for pw in pathways:
                for (
                    vname,
                    unit,
                    vlow,
                    vbase,
                    vhigh,
                    ltch_l,
                    ltch_h,
                ) in vars_map[pw]:
                    records.append({
                        "State": st_code,
                        "Pathway": pw,
                        "Variable Name": vname,
                        "Variable Unit": unit,
                        "Variable (low)": vlow,
                        "Variable (base)": vbase,
                        "Variable (high)": vhigh,
                        "LTCH Low": ltch_l,
                        "LTCH High": ltch_h,
                        "LTCH Base": (ltch_l + ltch_h) / 2.0,
                    })
        return pd.DataFrame(records)


@st.cache_data
def load_mc_data():
    try:
        df = pd.read_csv("MCDataFile.csv")
        return df
    except FileNotFoundError:
        states = ["TX", "CA", "AZ", "CO", "UT", "NV", "NM", "OK"]
        pathways = ["SMR", "SMRCC", "Electrolysis"]
        records = []
        np.random.seed(42)
        for st_code in states:
            for pw in pathways:
                base_val = 2.50 if pw == "SMR" else (2.80 if pw == "SMRCC" else 4.20)
                samples = np.random.normal(loc=base_val, scale=0.40, size=300)
                for val in samples:
                    records.append({
                        "State": st_code,
                        "Pathway": pw,
                        "LTCH": max(0.50, val),
                    })
        return pd.DataFrame(records)


@st.cache_data
def load_inputs_dataset():
    try:
        df = pd.read_csv("InputsDataset.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame()


@st.cache_data
def load_cng_stations():
    try:
        df = pd.read_csv("cng_stations_all_states.csv")
        return df
    except FileNotFoundError:
        return pd.DataFrame()


cng_df = load_cng_stations()

cng_lat_col = find_column(cng_df.columns, ["Latitude", "Lat", "LATITUDE"]) if not cng_df.empty else None
cng_lon_col = find_column(cng_df.columns, ["Longitude", "Lon", "Lng", "LONGITUDE"]) if not cng_df.empty else None
cng_price_col = find_column(
    cng_df.columns, 
    ["CNG Price ($/GGE)", "CNG Price ($)", "Price ($/GGE)", "CNG_Price_GGE", "CNG Price"]
) if not cng_df.empty else None
cng_state_col = find_column(
    cng_df.columns,
    ["state", "State", "STATE"]
) if not cng_df.empty else None

if not cng_df.empty and cng_price_col:
    cng_valid = cng_df.copy()
    
    # Strip currency symbols and whitespace if string
    if cng_valid[cng_price_col].dtype == object:
        cng_valid[cng_price_col] = (
            cng_valid[cng_price_col]
            .astype(str)
            .str.replace("$", "", regex=False)
            .str.replace(",", "", regex=False)
            .str.strip()
        )
    
    cng_valid[cng_price_col] = pd.to_numeric(cng_valid[cng_price_col], errors="coerce")
    cng_valid = cng_valid.dropna(subset=[cng_price_col])
    
    # Validate realistic CNG price range ($0.50 - $15.00/GGE)
    cng_valid = cng_valid[(cng_valid[cng_price_col] >= 0.50) & (cng_valid[cng_price_col] <= 15.00)]
    
    if cng_valid.empty:
        cng_price_col = None
else:
    cng_valid = pd.DataFrame()


def get_cng_price_by_closest_state(lat=None, lon=None, state_code=None, fallback_price=2.50):
    """
    Retrieves the CNG fuel price from cng_stations_all_states.csv matching the selected state code
    or geographically closest US state centroid.
    """
    if cng_valid.empty or cng_price_col is None:
        return fallback_price

    # Strategy 1: Match directly by state abbreviation or state name
    if state_code:
        st_abbr = normalize_state_str(state_code)
        if st_abbr and cng_state_col:
            state_df = cng_valid[
                cng_valid[cng_state_col].astype(str).apply(normalize_state_str) == st_abbr
            ]
            if not state_df.empty:
                avg_price = state_df[cng_price_col].mean()
                if not pd.isna(avg_price) and 0.50 <= avg_price <= 15.00:
                    return float(avg_price)

    # Strategy 2: Determine closest US state spatially using coordinates (lat, lon)
    if lat is not None and lon is not None and not (pd.isna(lat) or pd.isna(lon)):
        try:
            _, idx = state_centroid_tree.query([lat, lon])
            nearest_state_abbr = STATE_CENTROID_ABBRS[idx]
            if cng_state_col:
                state_df = cng_valid[
                    cng_valid[cng_state_col].astype(str).apply(normalize_state_str) == nearest_state_abbr
                ]
                if not state_df.empty:
                    avg_price = state_df[cng_price_col].mean()
                    if not pd.isna(avg_price) and 0.50 <= avg_price <= 15.00:
                        return float(avg_price)
        except Exception:
            pass

    return fallback_price


projects = load_projects()
tornado_df = load_tornado_data()
mc_df = load_mc_data()
inputs_df = load_inputs_dataset()

if projects.empty:
    st.error("Target dataset is empty or corrupted.")
    st.stop()


# ==========================================================
# GLOBAL COLUMN RESOLUTION
# ==========================================================
lat_col = find_column(projects.columns, ["Latitude", "Lat"])
lon_col = find_column(projects.columns, ["Longitude", "Lon", "Lng"])

if not lat_col or not lon_col:
    st.error(
        "Latitude and Longitude columns are missing or not recognizable in the dataset."
    )
    st.stop()

elec_col = find_column(
    projects.columns, ["Electricity Price ($/kWh)", "Electricity Price"]
)
ng_col_name = find_column(
    projects.columns, ["Natural Gas Price ($/MSCF)", "Natural Gas Price"]
)
water_col_name = find_column(
    projects.columns, ["Water Price ($/gallon)", "Water Price"]
)
co2_store_col = find_column(
    projects.columns, ["CO2 Storage Costs ($/kg H2)", "CO2 Storage Costs"]
)
co2_trans_col = find_column(
    projects.columns, ["CO2 Transport Costs ($/kg H₂)", "CO2 Transport Costs"]
)
gasoline_col = find_column(
    projects.columns, ["Gasoline Price ($/gal)", "Regular"]
)
diesel_col = find_column(
    projects.columns, ["Diesel Price ($/gal)", "Diesel Price", "Diesel"]
)
cng_col = find_column(
    projects.columns, ["CNG Price ($/GGE)", "CNG Price", "CNG ($/GGE)", "CNG"]
)

formation_col = find_column(
    projects.columns, ["Formation Name and Identifier", "Formation"]
)
lithology_col = find_column(projects.columns, ["Lithology"])
saline_col = find_column(
    projects.columns, ["Saline Aquifer Location", "Aquifer"]
)
op_col = find_column(projects.columns, ["Pipeline Operator", "Operator"])
type_col = find_column(projects.columns, ["Pipe Type", "Type"])
dist_col = find_column(projects.columns, ["Pipeline distances (mi)", "Distance"])

# Spatial KDTree Setup
coords_df = projects[[lat_col, lon_col]].dropna()
tree = cKDTree(coords_df.values)


# ==========================================================
# SESSION STATE & DEFAULT COORDINATE INIT
# ==========================================================

if "selected_index" not in st.session_state:
    st.session_state.selected_index = coords_df.index[0]

if "clicked_location" not in st.session_state:
    init_row = projects.loc[st.session_state.selected_index]
    st.session_state.clicked_location = (
        float(init_row[lat_col]),
        float(init_row[lon_col]),
    )

if "map_clicked" not in st.session_state:
    st.session_state.map_clicked = False

# --- PERSISTENT MAP LAYER STATE ---
if "show_saline" not in st.session_state:
    st.session_state.show_saline = True

if "show_pipeline" not in st.session_state:
    st.session_state.show_pipeline = True


# ==========================================================
# BUSINESS CASE MULTIPLIER DEFAULT CONFIGURATION
# ==========================================================

BUSINESS_CASE_DEFAULTS = {
    "Drayage": {
        "cng_low": 1.9,
        "cng_high": 3.6,
        "elec_low": 2.5,
        "elec_high": 5.4,
        "diesel_low": 1.6,
        "diesel_high": 3.0,
    },
    "Transit Bus": {
        "cng_low": 1.3,
        "cng_high": 2.5,
        "elec_low": 2.1,
        "elec_high": 5.0,
        "diesel_low": 1.0,
        "diesel_high": 1.9,
    },
    "Refuse Truck": {
        "cng_low": 1.1,
        "cng_high": 2.3,
        "elec_low": 2.6,
        "elec_high": 6.3,
        "diesel_low": 0.8,
        "diesel_high": 1.7,
    },
}


def update_multiplier_defaults():
    selected_bc = st.session_state.get("business_case", "Drayage")
    defaults = BUSINESS_CASE_DEFAULTS.get(
        selected_bc, BUSINESS_CASE_DEFAULTS["Drayage"]
    )
    for key, val in defaults.items():
        st.session_state[key] = val


if "business_case" not in st.session_state:
    st.session_state.business_case = "Drayage"
    update_multiplier_defaults()


# ==========================================================
# MAIN TITLE
# ==========================================================

st.title("National Hydrogen Transportation Fuel Supply")


# ==========================================================
# SIDEBAR VIEW AND MULTIPLIER INPUTS
# ==========================================================

st.sidebar.radio(
    "Business Case",
    ["Drayage", "Transit Bus", "Refuse Truck"],
    key="business_case",
    on_change=update_multiplier_defaults,
)

st.sidebar.markdown("---")
st.sidebar.subheader("Price Competitive Band Multipliers")

with st.sidebar.expander("Adjust Multipliers", expanded=True):
    st.markdown("**CNG (Compressed Natural Gas)**")
    cng_low = st.number_input(
        "CNG Low", step=0.05, format="%.3f", key="cng_low"
    )
    cng_high = st.number_input(
        "CNG High", step=0.05, format="%.3f", key="cng_high"
    )

    st.markdown("**Electricity**")
    elec_low = st.number_input(
        "Electricity Low", step=0.005, format="%.3f", key="elec_low"
    )
    elec_high = st.number_input(
        "Electricity High", step=0.005, format="%.3f", key="elec_high"
    )

    st.markdown("**Diesel**")
    diesel_low = st.number_input(
        "Diesel Low", step=0.05, format="%.3f", key="diesel_low"
    )
    diesel_high = st.number_input(
        "Diesel High", step=0.05, format="%.3f", key="diesel_high"
    )


# ==========================================================
# PROCESS CURRENT ROW SELECTION & STATE RESOLUTION
# ==========================================================

selected = projects.loc[st.session_state.selected_index]

state_col_proj = find_column(projects.columns, ["State_x"])
current_state = str(selected[state_col_proj]) if state_col_proj else "TX"


# ==========================================================
# LCOH COMPONENT TRACKING ENGINE UTILITY
# ==========================================================

display_order = [
    "SMR",
    "SMRCC",
    "Electrolysis",
    "Onsite SMR",
    "Onsite Electrolysis",
    "SMR Liquid",
    "SMRCC Liquid",
    "Electrolysis Liquid",
]

component_order = [
    "Electricity ($/kg H₂)",
    "Water ($/kg H₂)",
    "Natural Gas ($/kg H₂)",
    "Variable Operating Costs ($/kg H₂)",
    "Operating and Maintenance ($/kg H₂)",
    "Capital Costs ($/kg H₂)",
    "CO₂ Storage Costs ($/kg H₂)",
    "CO₂ Transport Costs ($/kg H₂)",
    "Levelized Cost of Transport (LCT) ($/kg H₂)",
    "Levelized Cost of Refueling Stations (LCRS) ($/kg H₂)",
]


def calculate_row_smrcc_total(row_data, columns):
    total = 0.0
    has_any = False
    for col in columns:
        col_upper = col.upper()
        comp_display = None

        if any(k in col_upper for k in ["LFCKWH"]):
            comp_display = "Electricity"
        elif any(k in col_upper for k in ["LFC O", "LFCH2O"]):
            comp_display = "Water"
        elif any(k in col_upper for k in ["LFCC 4", "LFCCH4"]):
            comp_display = "Natural Gas"
        elif any(k in col_upper for k in ["LCOV", "VARIABLE OPERATING"]):
            comp_display = "Variable"
        elif any(k in col_upper for k in ["LOM", "OPERATING", "O&M"]):
            comp_display = "OM"
        elif "LCC" in col_upper or "CAPITAL" in col_upper:
            comp_display = "Capital"
        elif "CO2" in col_upper and "STORAGE" in col_upper:
            comp_display = "Storage"
        elif "CO2" in col_upper and "TRANSPORT" in col_upper:
            comp_display = "Transport"
        elif "LCT" in col_upper or (
            "TRANSPORT" in col_upper and "CO2" not in col_upper
        ):
            comp_display = "LCT"
        elif any(k in col_upper for k in ["LCRS", "REFUELING", "STATION"]):
            comp_display = "LCRS"

        if not comp_display or any(
            ex in col_upper for ex in ["LCO ", "LCCO", "LFC "]
        ):
            continue

        if "LIQUID" in col_upper:
            continue

        matched_techs = []
        for tech in [
            "Onsite SMR",
            "Onsite Electrolysis",
            "SMRCC",
            "Electrolysis",
            "SMR",
        ]:
            if tech.lower() in col.lower():
                matched_techs.append(tech)
                break

        if not matched_techs or "SMRCC" in matched_techs:
            try:
                val = float(row_data[col])
                if not np.isnan(val):
                    total += val
                    has_any = True
            except Exception:
                pass
    return total if has_any else None


has_parsed_data = any(
    any(k in c.upper() for k in ["LFCKWH", "LFCH2O", "LFCCH4", "LCC", "LOM"])
    for c in projects.columns
)


# ==========================================================
# MAIN TABBED INTERFACE LAYOUT
# ==========================================================

tab_overview, tab_co2, tab_pathway, tab_values, tab_external = st.tabs([
    "Home",
    "CO₂ Emissions",
    "Analysis",
    "Values",
    "Full TCO Model",
])


# ==========================================================
# TAB 1: MAP AND OVERVIEW DASHBOARD
# ==========================================================

with tab_overview:
    table_col, map_col = st.columns([1.8, 3.3])

    # 1. SIDE METADATA TABLE
    with table_col:
        if st.session_state.map_clicked:
            city_col = find_column(
                projects.columns, ["Name_right", "City", "Site"]
            )
            state_col = find_column(projects.columns, [])

            if city_col and state_col:
                city_state_val = f"{selected[city_col]}, {selected[state_col]}"
            elif city_col:
                city_state_val = str(selected[city_col])
            elif state_col:
                city_state_val = str(selected[state_col])
            else:
                city_state_val = "N/A"

            val_elec = (
                float(selected[elec_col])
                if elec_col and not pd.isna(selected[elec_col])
                else 0.0
            )
            val_ng = (
                float(selected[ng_col_name])
                if ng_col_name and not pd.isna(selected[ng_col_name])
                else 0.0
            )
            val_water = (
                float(selected[water_col_name])
                if water_col_name and not pd.isna(selected[water_col_name])
                else 0.0
            )
            val_co2_s = (
                float(selected[co2_store_col])
                if co2_store_col and not pd.isna(selected[co2_store_col])
                else 0.0
            )
            val_co2_t = (
                float(selected[co2_trans_col])
                if co2_trans_col and not pd.isna(selected[co2_trans_col])
                else 0.0
            )
            val_gas = (
                float(selected[gasoline_col])
                if gasoline_col and not pd.isna(selected[gasoline_col])
                else 0.0
            )
            val_diesel = (
                float(selected[diesel_col])
                if diesel_col and not pd.isna(selected[diesel_col])
                else 0.0
            )

            fallback_cng = float(selected[cng_col]) if cng_col and not pd.isna(selected[cng_col]) else 2.50
            selected_lat = float(selected[lat_col])
            selected_lon = float(selected[lon_col])

            val_cng = get_cng_price_by_closest_state(
                selected_lat, 
                selected_lon, 
                state_code=current_state, 
                fallback_price=fallback_cng
            )

            calc_co2_tonne = 100.0 * (val_co2_s + val_co2_t)

            matched_table_rows = [
                ("Nearest City, State", city_state_val),
                ("Electricity Price ($/kWh)", fmt_money(val_elec)),
                ("Natural Gas Price ($/MSCF)", fmt_money(val_ng)),
                ("Water Price ($/gallon)", fmt_money(val_water)),
                (
                    "CO<sub>2</sub> Storage and Transport Costs ($/ tonne CO<sub>2</sub>)",
                    fmt_money(calc_co2_tonne),
                ),
                (
                    "CO<sub>2</sub> Storage Cost ($/kg H<sub>2</sub>)",
                    fmt_money(val_co2_s),
                ),
                (
                    "CO<sub>2</sub> Transport Costs ($/kg H<sub>2</sub>)",
                    fmt_money(val_co2_t),
                ),
                ("Gasoline Price ($/gal)", fmt_money(val_gas)),
                ("Diesel Price ($/gal)", fmt_money(val_diesel)),
                ("CNG Price ($/GGE)", fmt_money(val_cng)),
            ]

            html_table = "<table style='width:100%; border-collapse: collapse;'>"
            for idx_row, (label, val_str) in enumerate(matched_table_rows):
                bg = "#F5F5F5" if idx_row % 2 == 1 else "white"
                html_table += (
                    f"<tr style='background-color: {bg};'>"
                    f"<td style='padding: 15px 25px; text-align: center; font-size: 10.5pt; font-weight: none; line-height:2.3; border: none;'>{label}</td>"
                    f"<td style='padding: 15px 25px; text-align: center; font-size: 10.5pt; border: none; white-space: nowrap;'>{val_str}</td>"
                    f"</tr>"
                )
            html_table += "</table>"
            st.html(html_table)
        else:
            st.markdown(
                """
                <div style="
                    border: None; 
                    border-radius: 8px; 
                    padding: 40px 20px; 
                    text-align: center; 
                    color: #666666; 
                    font-family: Arial, sans-serif;
                    font-size: 11pt;
                    margin-top: 10px;
                ">
                    <p style="font-weight: bold; margin-bottom: 8px;"></p>
                    <p style="margin: 0;"></p>
                </div>
                """,
                unsafe_allow_html=True,
            )

# 2. FOLIUM GEOGRAPHIC MAP LAYER
    with map_col:
        m = folium.Map(
            location=[37.8283, -97.0795],
            zoom_start=5,
            control_scale=False,
            attribution_control=False,
        )

        folium.TileLayer(
            tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Topo_Map/MapServer/tile/{z}/{y}/{x}",
            attr="&copy; OpenStreetMap &copy; CARTO",
            name="Tiles © Esri — DeLorme",
            overlay=False,
            control=False,
        ).add_to(m)

        # 2a. Saline Aquifers Layer (Blue-Green filled, 30% Opacity)
        saline_aquifers = load_saline_aquifers_layer()
        if saline_aquifers:
            folium.GeoJson(
                saline_aquifers,
                name="NATCARB Saline Aquifers",
                style_function=saline_style,
                show=st.session_state.show_saline,  # <--- Persisted state
            ).add_to(m)

        # 2b. Natural Gas Pipeline Layer
        pipeline = load_pipeline_layer()
        if pipeline:
            folium.GeoJson(
                pipeline,
                name="Natural Gas Pipelines",
                style_function=pipeline_style,
                show=st.session_state.show_pipeline,  # <--- Persisted state
            ).add_to(m)

        avg_diesel_m = (diesel_low + diesel_high) / 2.0
        avg_cng_m = (cng_low + cng_high) / 2.0
        avg_elec_m = (elec_low + elec_high) / 2.0

        for idx, row in projects.iterrows():
            lat_val = float(row[lat_col])
            lon_val = float(row[lon_col])

            if np.isnan(lat_val) or np.isnan(lon_val):
                continue

            if has_parsed_data:
                smrcc_total = (
                    calculate_row_smrcc_total(row, projects.columns) or 2.50
                )
            else:
                np.random.seed(int(idx) % 100)
                smrcc_total = sum(
                    np.random.uniform(0.1, 0.4, len(component_order))
                )

            v_elec = (
                float(row[elec_col])
                if elec_col and not pd.isna(row[elec_col])
                else 0.0
            )
            v_ng = (
                float(row[ng_col_name])
                if ng_col_name and not pd.isna(row[ng_col_name])
                else 0.0
            )
            v_water = (
                float(row[water_col_name])
                if water_col_name and not pd.isna(row[water_col_name])
                else 0.0
            )
            v_co2_s = (
                float(row[co2_store_col])
                if co2_store_col and not pd.isna(row[co2_store_col])
                else 0.0
            )
            v_co2_t = (
                float(row[co2_trans_col])
                if co2_trans_col and not pd.isna(row[co2_trans_col])
                else 0.0
            )
            v_gas = (
                float(row[gasoline_col])
                if gasoline_col and not pd.isna(row[gasoline_col])
                else 0.0
            )
            v_diesel = (
                float(row[diesel_col])
                if diesel_col and not pd.isna(row[diesel_col])
                else 0.0
            )

            fallback_cng_row = float(row[cng_col]) if cng_col and not pd.isna(row[cng_col]) else 2.50

            row_state = str(row[state_col_proj]) if state_col_proj else None
            v_cng = get_cng_price_by_closest_state(
                lat_val, 
                lon_val, 
                state_code=row_state, 
                fallback_price=fallback_cng_row
            )

            val_form = str(row[formation_col]) if formation_col else "N/A"
            val_lith = str(row[lithology_col]) if lithology_col else "N/A"
            val_saline = str(row[saline_col]) if saline_col else "N/A"
            val_op = str(row[op_col]) if op_col else "N/A"
            val_type = str(row[type_col]) if type_col else "N/A"

            val_dist = "N/A"
            if dist_col and not pd.isna(row[dist_col]):
                try:
                    val_dist = f"{float(row[dist_col]):.1f} mi"
                except ValueError:
                    val_dist = str(row[dist_col])

            ratio_diesel = (
                f"{smrcc_total / (v_diesel * avg_diesel_m):.2f}"
                if v_diesel > 0
                else "N/A"
            )
            ratio_cng = (
                f"{smrcc_total / (v_cng * avg_cng_m):.2f}"
                if v_cng > 0
                else "N/A"
            )
            ratio_elec = (
                f"{smrcc_total / (v_elec * avg_elec_m):.2f}"
                if v_elec > 0
                else "N/A"
            )

            co2_combined_calc = (v_co2_s + v_co2_t) * 100.0
            co2_comb_str = (
                f"${co2_combined_calc:.2f}" if co2_combined_calc > 0 else "N/A"
            )

            tooltip_html = f"""
            <div style="font-family: 'Arial', sans-serif; font-size: 11pt; line-height: 1.45; color: #333333; min-width: 325px; padding: 5px;">
                <b>Competitiveness ratio SMRCC to diesel:</b> {ratio_diesel}<br>
                <b>Competitiveness ratio SMRCC to CNG:</b> {ratio_cng}<br>
                <b>Competitiveness ratio SMRCC to Electricity:</b> {ratio_elec}<br>
                <br>
                <b>Electricity price ($/kWh):</b> {fmt_money(v_elec)}<br>
                <b>Natural Gas price ($/MSCF):</b> {fmt_money(v_ng)}<br>
                <b>Water price ($/gallon):</b> {fmt_money(v_water)}<br>
                <br>
                <b>CO<sub>2</sub> storage and transport costs ($/ tonne CO<sub>2</sub>):</b> {co2_comb_str}<br>
                <b>CO<sub>2</sub> storage cost ($/kg H<sub>2</sub>):</b> {fmt_money(v_co2_s)}<br>
                <b>CO<sub>2</sub> transport costs ($/kg H<sub>2</sub>):</b> {fmt_money(v_co2_t)}<br>
                <b>Formation Name and Identifier:</b> {val_form}<br>
                <b>Lithology:</b> {val_lith}<br>
                <b>Saline Aquifer Location:</b> {val_saline}<br>
                <br>
                <b>Gasoline price ($/gal):</b> {fmt_money(v_gas)}<br>
                <b>Diesel price ($/gal):</b> {fmt_money(v_diesel)}<br>
                <b>CNG price ($/GGE):</b> {fmt_money(v_cng)}<br>
                <br>
                <b>Pipeline operator:</b> {val_op}<br>
                <b>Pipe type:</b> {val_type}<br>
                <b>Pipeline distances (mi):</b> {val_dist}
            </div>
            """

            # Invisible Square Bounding Box around location for hover/tooltip
            sq_offset = 1  # Degree offset for square size
            square_bounds = [
                [lat_val - sq_offset, lon_val - sq_offset],
                [lat_val + sq_offset, lon_val + sq_offset],
            ]

            folium.Rectangle(
                bounds=square_bounds,
                color="rgba(0,0,0,0)",
                weight=0,
                fill=True,
                fill_color="rgba(0,0,0,0)",
                fill_opacity=0.0,
                interactive=True,
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
            ).add_to(m)

        if (
            st.session_state.map_clicked
            and st.session_state.clicked_location is not None
        ):
            folium.Marker(
                location=st.session_state.clicked_location,
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)

        folium.LayerControl().add_to(m)

        event = st_folium(
            m,
            height=740,
            use_container_width=True,
            returned_objects=["last_clicked"],
            key="hydrogen_map",
        )

        if event and event.get("last_clicked"):
            lat = event["last_clicked"]["lat"]
            lon = event["last_clicked"]["lng"]
            new_click = (lat, lon)

            if (
                st.session_state.clicked_location != new_click
                or not st.session_state.map_clicked
            ):
                st.session_state.clicked_location = new_click
                _, idx_in_coords = tree.query([lat, lon])
                st.session_state.selected_index = int(
                    coords_df.index[idx_in_coords]
                )
                st.session_state.map_clicked = True
                st.rerun()

    # 3. LCOH STACKED BAR CHART
    if st.session_state.map_clicked:
        component_colors = [
            "rgba(52, 58, 64, 0.70)",
            "rgba(72, 202, 228, 0.75)",
            "rgba(220, 53, 69, 0.75)",
            "rgba(153, 102, 255, 0.70)",
            "rgba(0, 180, 216, 0.85)",
            "rgba(0, 123, 255, 0.85)",
            "rgba(111, 66, 193, 0.75)",
            "rgba(253, 126, 20, 0.80)",
            "rgba(108, 117, 125, 0.75)",
            "rgba(255, 193, 7, 0.80)",
        ]
        color_lookup = {
            comp: component_colors[i % len(component_colors)]
            for i, comp in enumerate(component_order)
        }

        fig = go.Figure()

        v_elec = (
            float(selected[elec_col])
            if elec_col and not pd.isna(selected[elec_col])
            else 0.0
        )

        fallback_chart_cng = float(selected[cng_col]) if cng_col and not pd.isna(selected[cng_col]) else 2.50

        v_cng = get_cng_price_by_closest_state(
            float(selected[lat_col]), 
            float(selected[lon_col]), 
            state_code=current_state, 
            fallback_price=fallback_chart_cng
        )

        v_diesel = (
            float(selected[diesel_col])
            if diesel_col and not pd.isna(selected[diesel_col])
            else 0.0
        )

        if has_parsed_data:
            tech_values = {t: {} for t in display_order}
            components_set = set()
            techs_present = set()

            for col in projects.columns:
                col_upper = col.upper()
                comp_display = None

                is_liquid_lct = "LIQUID" in col_upper and (
                    "LCT" in col_upper
                    or ("TRANSPORT" in col_upper and "CO2" not in col_upper)
                )
                is_liquid_lcrs = "LIQUID" in col_upper and any(
                    k in col_upper for k in ["LCRS", "REFUELING", "STATION"]
                )
                is_gas_lct = "LIQUID" not in col_upper and (
                    "LCT" in col_upper
                    or ("TRANSPORT" in col_upper and "CO2" not in col_upper)
                )
                is_gas_lcrs = "LIQUID" not in col_upper and any(
                    k in col_upper for k in ["LCRS", "REFUELING", "STATION"]
                )

                if any(k in col_upper for k in ["LFCKWH"]):
                    comp_display = "Electricity ($/kg H₂)"
                elif any(k in col_upper for k in ["LFC O", "LFCH2O"]):
                    comp_display = "Water ($/kg H₂)"
                elif any(k in col_upper for k in ["LFCC 4", "LFCCH4"]):
                    comp_display = "Natural Gas ($/kg H₂)"
                elif any(k in col_upper for k in ["LCOV", "VARIABLE OPERATING"]):
                    comp_display = "Variable Operating Costs ($/kg H₂)"
                elif any(
                    k in col_upper
                    for k in ["LOM", "OPERATING AND MAINTENANCE", "O&M", "O & M"]
                ):
                    comp_display = "Operating and Maintenance ($/kg H₂)"
                elif "LCC" in col_upper or "CAPITAL" in col_upper:
                    comp_display = "Capital Costs ($/kg H₂)"
                elif "CO2" in col_upper and "STORAGE" in col_upper:
                    comp_display = "CO₂ Storage Costs ($/kg H₂)"
                elif "CO2" in col_upper and "TRANSPORT" in col_upper:
                    comp_display = "CO₂ Transport Costs ($/kg H₂)"
                elif is_liquid_lct or is_gas_lct:
                    comp_display = "Levelized Cost of Transport (LCT) ($/kg H₂)"
                elif is_liquid_lcrs or is_gas_lcrs:
                    comp_display = (
                        "Levelized Cost of Refueling Stations (LCRS) ($/kg H₂)"
                    )

                if not comp_display or any(
                    ex in col_upper for ex in ["LCO ", "LCCO", "LFC "]
                ):
                    continue

                matched_techs = []
                col_lower = col.lower()
                if "onsite electrolysis" in col_lower:
                    matched_techs = ["Onsite Electrolysis"]
                elif "onsite smr" in col_lower:
                    matched_techs = ["Onsite SMR"]
                elif "smrcc" in col_lower:
                    if "liquid" in col_lower:
                        matched_techs = ["SMRCC Liquid"]
                    else:
                        if is_liquid_lct or is_liquid_lcrs:
                            matched_techs = ["SMRCC Liquid"]
                        elif is_gas_lct or is_gas_lcrs:
                            matched_techs = ["SMRCC"]
                        else:
                            matched_techs = ["SMRCC", "SMRCC Liquid"]
                elif "electrolysis" in col_lower:
                    if "liquid" in col_lower:
                        matched_techs = ["Electrolysis Liquid"]
                    else:
                        if is_liquid_lct or is_liquid_lcrs:
                            matched_techs = ["Electrolysis Liquid"]
                        elif is_gas_lct or is_gas_lcrs:
                            matched_techs = ["Electrolysis"]
                        else:
                            matched_techs = ["Electrolysis", "Electrolysis Liquid"]
                elif "smr" in col_lower:
                    if "liquid" in col_lower:
                        matched_techs = ["SMR Liquid"]
                    else:
                        if is_liquid_lct or is_liquid_lcrs:
                            matched_techs = ["SMR Liquid"]
                        elif is_gas_lct or is_gas_lcrs:
                            matched_techs = ["SMR"]
                        else:
                            matched_techs = ["SMR", "SMR Liquid"]
                else:
                    if is_liquid_lct or is_liquid_lcrs:
                        matched_techs = [
                            "SMR Liquid",
                            "SMRCC Liquid",
                            "Electrolysis Liquid",
                        ]
                    elif is_gas_lct or is_gas_lcrs:
                        matched_techs = [
                            "SMR",
                            "SMRCC",
                            "Electrolysis",
                            "Onsite SMR",
                            "Onsite Electrolysis",
                        ]
                    else:
                        matched_techs = display_order

                try:
                    val = float(selected[col])
                    if np.isnan(val):
                        val = 0.0
                except Exception:
                    val = 0.0

                for t_name in matched_techs:
                    is_lct = (
                        comp_display == "Levelized Cost of Transport (LCT) ($/kg H₂)"
                    )
                    is_onsite = t_name in ["Onsite SMR", "Onsite Electrolysis"]
                    val_to_add = 0.0 if (is_onsite and is_lct) else val

                    tech_values[t_name][comp_display] = (
                        tech_values[t_name].get(comp_display, 0.0) + val_to_add
                    )
                    techs_present.add(t_name)
                components_set.add(comp_display)

            ordered_techs = [t for t in display_order if t in techs_present]
            ordered_components = [c for c in component_order if c in components_set]
            for c in components_set:
                if c not in ordered_components:
                    ordered_components.append(c)

            for comp in reversed(ordered_components):
                y_vals = [tech_values[t].get(comp, 0.0) for t in ordered_techs]
                fig.add_trace(
                    go.Bar(
                        x=ordered_techs,
                        y=y_vals,
                        name=comp,
                        textfont=dict(size=22, color="black"),               
                        marker_color=color_lookup.get(
                            comp, "rgba(128,128,128,0.60)"
                        ),
                    )
                )

            totals = [sum(tech_values[t].values()) for t in ordered_techs]
            fig.add_trace(
                go.Bar(
                    x=ordered_techs,
                    y=[0] * len(ordered_techs),
                    name="Total LCOH",
                    showlegend=False,
                    text=[f"${v:.2f}" for v in totals],
                    textfont=dict(size=22, color="#A9A9A9"),
                    textposition="outside",
                    cliponaxis=False,
                    hoverinfo="skip",
                )
            )
            fig.update_layout(legend_traceorder="reversed")

        else:
            np.random.seed(int(st.session_state.selected_index) % 100)
            scenarios = [t for t in display_order]
            mock_data = {
                comp: np.random.uniform(0.1, 0.4, len(scenarios))
                for comp in component_order
            }

            if "Levelized Cost of Transport (LCT) ($/kg H₂)" in mock_data:
                for t_name in ["Onsite SMR", "Onsite Electrolysis"]:
                    if t_name in scenarios:
                        mock_data["Levelized Cost of Transport (LCT) ($/kg H₂)"][
                            scenarios.index(t_name)
                        ] = 0.0

            totals = np.zeros(len(scenarios))

            for comp in reversed(component_order):
                vals = mock_data[comp]
                totals += vals
                fig.add_trace(
                    go.Bar(
                        x=scenarios, y=vals, name=comp, textfont=dict(size=22, color="black"),marker_color=color_lookup[comp]
                    )
                )

            fig.add_trace(
                go.Bar(
                    x=scenarios,
                    y=[0] * len(scenarios),
                    name="Total LCOH",
                    showlegend=False,
                    text=[f"${v:.2f}" for v in totals],
                    font=dict(size=20, color="#1A1A1A"),
                    textposition="outside",
                    cliponaxis=False,
                )
            )
            fig.update_layout(legend_traceorder="reversed")

        fuel_bands = [
            (
                "Price-Competitive Band - CNG",
                v_cng * cng_low,
                v_cng * cng_high,
                "rgba(0,255,0,0.25)",
            ),
            (
                "Price-Competitive Band - Electricity",
                v_elec * elec_low,
                v_elec * elec_high,
                "rgba(255,193,7,0.30)",
            ),
            (
                "Price-Competitive Band - Diesel",
                v_diesel * diesel_low,
                v_diesel * diesel_high,
                "rgba(220,0,0,0.25)",
            ),
        ]

        for label, band_low, band_high, color in fuel_bands:
            fig.add_hrect(
                y0=band_low,
                y1=band_high,
                fillcolor=color,
                line_width=0,
                layer="below",
            )
            fig.add_trace(
                go.Scatter(
                    x=[None],
                    y=[None],
                    mode="markers",
                    marker=dict(size=14, color=color),
                    name=label,
                )
            )

        fig.update_layout(
            barmode="stack",
            height=550,
            template="plotly_white",
            xaxis=dict(
                side="top",
                title=dict(
                    text="",
                    font=dict(size=22, family="Calibri Bold")
                ),tickfont=dict(size=22, color="black")
            ),
            yaxis=dict(
                title=dict(
                    text="Levelized Cost of Hydrogen [$ / kg H₂]",
                    font=dict(size=22,color="#000000")
                ),tickfont=dict(size=22)
            ),

            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.02,
                font=dict(size=22), title=dict(
                    text="Legend",
                    font=dict(size=22,color="#000000"),
                )
            ),
            margin=dict(l=50, r=150, t=60, b=50),
        )

        st.plotly_chart(fig, use_container_width=True)


# ==========================================================
# TAB 2: CO2 EMISSIONS ANALYSIS
# ==========================================================

with tab_co2:
    st.subheader("")

    # Locate BEV and ICEV emission columns from selected row
    bev_co2_col = find_column(
        projects.columns,
        ["BEV CO2 Emissions (lb CO2/kWh)", "BEV CO2 Emissions", "BEV CO2", "BEV"]
    )
    icev_co2_col = find_column(
        projects.columns,
        ["ICEV CO2 Emissions per mile (kg/mile)", "ICEV CO2 Emissions", "ICEV CO2", "ICEV"]
    )

    val_bev_co2 = (
        float(selected[bev_co2_col])
        if bev_co2_col and not pd.isna(selected[bev_co2_col])
        else 0.85
    )
    val_icev_co2 = (
        float(selected[icev_co2_col])
        if icev_co2_col and not pd.isna(selected[icev_co2_col])
        else 1.45
    )

    # Pathway Calculations
    pathway_names = [
        "SMR",
        "SMRCC",
        "Electrolysis",
        "Onsite SMR",
        "Onsite Electrolysis",
    ]

    smr_gen = 0.3 / 2.2
    smrcc_gen = (1.5 / 67.0) * val_bev_co2 / 2.2
    elec_gen = (55.5 / 67.0) * val_bev_co2 / 2.2
    onsite_smr_gen = 0.3 / 2.2
    onsite_elec_gen = (55.5 / 67.0) * val_bev_co2 / 2.2

    h2_gen_emissions = [
        smr_gen,
        smrcc_gen,
        elec_gen,
        onsite_smr_gen,
        onsite_elec_gen,
    ]

    refueling_station_emissions_val = (5.16 / 67.0) * val_bev_co2 / 2.2
    refueling_emissions = [refueling_station_emissions_val] * 5

    total_pathway_emissions = [
        g + r for g, r in zip(h2_gen_emissions, refueling_emissions)
    ]

    # 1. Emissions Summary Table
    st.markdown("### Emissions Summary Table")

    df_emissions = pd.DataFrame({
        "Pathway": pathway_names,
        "Hydrogen Generation Emissions [kg CO₂ / mi]": [
            f"{val:.4f}" for val in h2_gen_emissions
        ],
        "Refueling Station Emissions [kg CO₂ / mi]": [
            f"{val:.4f}" for val in refueling_emissions
        ],
        "Total Emissions [kg CO₂ / mi]": [
            f"{val:.4f}" for val in total_pathway_emissions
        ],
    })

    st.dataframe(df_emissions, use_container_width=True, hide_index=True)

    st.markdown("---")

    # 2. Stacked Bar Chart with ICEV Line
    fig_co2_chart = go.Figure()

     # Orange Stacked Bar: Refueling Station Emissions
    fig_co2_chart.add_trace(
        go.Bar(
            x=pathway_names,
            y=refueling_emissions,
            name="Refueling Station Emissions",
            marker_color="#FF8C00",
            hovertemplate="<b>%{x}</b><br>Refueling Station Emissions: %{y:.4f} kg CO₂/mi<extra></extra>",
        )
    )

   # Blue Stacked Bar: Hydrogen Generation
    fig_co2_chart.add_trace(
        go.Bar(
            x=pathway_names,
            y=h2_gen_emissions,
            name="Hydrogen Generation",
            marker_color="#0066FF",
            hovertemplate="<b>%{x}</b><br>Generation Emissions: %{y:.4f} kg CO₂/mi<extra></extra>",
        )
    )


    # Grey Line: ICEV Emissions
    fig_co2_chart.add_trace(
        go.Scatter(
            x=pathway_names,
            y=[val_icev_co2] * len(pathway_names),
            mode="lines",
            name="ICEV Emissions",
            line=dict(color="#707070", width=3),
            hovertemplate="<b>ICEV Emissions</b>: %{y:.4f} kg CO₂/mi<extra></extra>",
        )
    )

    # Add Black Bold Labels above each stacked bar
    for idx_p, p_name in enumerate(pathway_names):
        tot_val = total_pathway_emissions[idx_p]
        fig_co2_chart.add_annotation(
            x=p_name,
            y=tot_val,
            text=f"<b>{tot_val:.3f}</b>",
            showarrow=False,
            yshift=12,
            font=dict(color="#000000", size=14, family="Arial Black"),
        )

    max_y_val = max(max(total_pathway_emissions), val_icev_co2) * 1.25

    fig_co2_chart.update_layout(
        barmode="stack",
        height=550,
        template="plotly_white",
        showlegend=True,
        xaxis=dict(
            side="top",
            title=dict(
                text="",
                font=dict(size=18, family="Arial Bold", color="#000000"),
            ),
            tickfont=dict(size=14, color="#000000"),
        ),
        yaxis=dict(
            title=dict(
                text="Emissions per Mile [kg CO₂ / mi]",
                font=dict(size=20, family="Arial Bold", color="#000000"),
            ),
            range=[0, max_y_val],
            tickfont=dict(size=14, color="#000000"),
        ),
        legend=dict(
            orientation="v",
            yanchor="top",
            y=1,
            xanchor="left",
            x=1.02,
            font=dict(size=18), title=dict(
                text="Legend",
                font=dict(size=22,color="#000000"),
            )
        ),
        margin=dict(l=60, r=40, t=80, b=50),
    )

    st.plotly_chart(fig_co2_chart, use_container_width=True)


# ==========================================================
# TAB 3: PATHWAY SENSITIVITY & MONTE CARLO ANALYSIS
# ==========================================================

with tab_pathway:
    st.subheader(f"")

    # Pathway Selector
    selected_pathway = st.radio(
        "Select H₂ Generation Pathway:",
        ["SMR", "SMRCC", "Electrolysis"],
        horizontal=True,
    )

    # Resolve LCT and LCRS costs for LCOH calculation
    lcrs_col = find_column(
        projects.columns,
        ["Levelized Cost of Refueling Stations", "LCRS", "Refueling"],
    )
    lct_col = find_column(
        projects.columns, ["Levelized Cost of Transport", "LCT", "Transport"]
    )

    cost_lcrs = (
        float(selected[lcrs_col])
        if lcrs_col and not pd.isna(selected[lcrs_col])
        else 1.25
    )
    cost_lct = (
        float(selected[lct_col])
        if lct_col and not pd.isna(selected[lct_col])
        else 0.50
    )
    deduction = cost_lcrs + cost_lct

    # ==========================================================
    # 1. RESOLVE COLUMNS & FILTER TORNADO SENSITIVITY DATA
    # ==========================================================
    state_col_torn = find_column(tornado_df.columns, ["State", "ST"])
    path_col_torn = find_column(tornado_df.columns, ["Pathway", "Technology"])

    filter_torn = tornado_df.copy()
    if state_col_torn:
        filter_torn = filter_torn[
            filter_torn[state_col_torn].astype(str).str.upper()
            == current_state.upper()
        ]
    if path_col_torn:
        filter_torn = filter_torn[
            filter_torn[path_col_torn].astype(str).str.upper()
            == selected_pathway.upper()
        ]

    ltch_low_col = find_column(
        filter_torn.columns,
        ["Result (low)", "LTCH Low", "LTCH low", "LTCH (low)", "LTCH_low", "ltch_low", "LTCH Min"],
    )
    ltch_high_col = find_column(
        filter_torn.columns,
        ["Result (high)", "LTCH High", "LTCH high", "LTCH (high)", "LTCH_high", "ltch_high", "LTCH Max"],
    )
    ltch_base_col = find_column(
        filter_torn.columns,
        [
            "Result (base)",
            "Result Base",
            "LTCH Base",
            "LTCH base",
            "LTCH (base)",
            "LTCH_base",
            "ltch_base",
            "LTCH Mid",
        ],
    )
    var_name_col = find_column(
        filter_torn.columns, ["Variable Name", "Variable", "Parameter", "Name"]
    )

    # Clean empty strings / nulls
    filter_torn = filter_torn.replace(r"^\s*$", np.nan, regex=True)
    filter_torn = filter_torn.replace(["nan", "NaN", "None", "N/A", "n/a"], np.nan)

    if var_name_col and var_name_col in filter_torn.columns:
        filter_torn = filter_torn.dropna(subset=[var_name_col])

    required_cols = [c for c in [ltch_low_col, ltch_high_col] if c and c in filter_torn.columns]
    if required_cols:
        filter_torn = filter_torn.dropna(subset=required_cols, how="any")

    if not filter_torn.empty and ltch_low_col and ltch_high_col:
        ltch_l_temp = pd.to_numeric(filter_torn[ltch_low_col], errors="coerce").values
        ltch_h_temp = pd.to_numeric(filter_torn[ltch_high_col], errors="coerce").values
        swings = np.abs(ltch_h_temp - ltch_l_temp)
        sort_idx = np.argsort(swings)[::-1]
        filter_torn = filter_torn.iloc[sort_idx].reset_index(drop=True)

    # ==========================================================
    # 2. SIDE-BY-SIDE LAYOUT (TABLE + TORNADO CHART)
    # ==========================================================
    col_table, col_chart = st.columns([1, 1.3])

    with col_table:
        st.markdown("")

        var_unit_col = find_column(
            filter_torn.columns, ["Variable Unit", "Unit", "Units"]
        )
        var_low_col = find_column(
            filter_torn.columns,
            ["Variable (low)", "Variable Low", "Var Low", "Low"],
        )
        var_base_col = find_column(
            filter_torn.columns,
            ["Variable (base)", "Variable Base", "Var Base", "Base"],
        )
        var_high_col = find_column(
            filter_torn.columns,
            ["Variable (high)", "Variable High", "Var High", "High"],
        )

        display_cols = [
            var_name_col,
            var_unit_col,
            var_low_col,
            var_base_col,
            var_high_col,
        ]
        avail_cols = [c for c in display_cols if c and c in filter_torn.columns]

        if not filter_torn.empty and avail_cols:
            df_display = filter_torn[avail_cols].dropna(subset=[var_name_col])

            # Build HTML Table with increased font size and empty spacer rows between entries
            html_analysis_table = "<table style='width:100%; border-collapse: collapse; margin-top: 10px;'>"
            html_analysis_table += "<tr style='background-color: #f2f2f2; border-bottom: 2px solid #ccc;'>"
            for col_name in df_display.columns:
                html_analysis_table += f"<th style='padding: 12px 10px; text-align: center; font-size: 16pt; font-weight: bold;'>{col_name}</th>"
            html_analysis_table += "</tr>"

            num_rows = len(df_display)
            for i, (_, row) in enumerate(df_display.iterrows()):
                bg = "#FFFFFF" if i % 2 == 0 else "#F9F9F9"
                html_analysis_table += f"<tr style='background-color: {bg};'>"
                for val in row:
                    val_str = "" if pd.isna(val) else str(val)
                    html_analysis_table += f"<td style='padding: 14px 10px; text-align: center; font-size: 15pt;'>{val_str}</td>"
                html_analysis_table += "</tr>"

                # Insert row (blank spacer row) between entries
                if i < num_rows - 1:
                    html_analysis_table += f"<tr style='height: 45px; background-color: transparent;'><td colspan='{len(df_display.columns)}' style='border: none;'></td></tr>"

            html_analysis_table += "</table>"
            st.html(html_analysis_table)
        else:
            st.info("No variable sensitivity data found for this state/pathway.")

    with col_chart:
        st.markdown("")

        if (
            not filter_torn.empty
            and ltch_low_col
            and ltch_high_col
            and var_name_col
        ):
            var_names = filter_torn[var_name_col].astype(str).values
            ltch_low = pd.to_numeric(filter_torn[ltch_low_col], errors="coerce").values
            ltch_high = pd.to_numeric(filter_torn[ltch_high_col], errors="coerce").values

            if ltch_base_col and ltch_base_col in filter_torn.columns:
                ltch_base = pd.to_numeric(filter_torn[ltch_base_col], errors="coerce").values
            else:
                ltch_base = (ltch_low + ltch_high) / 2.0

            lcoh_low = ltch_low - deduction
            lcoh_high = ltch_high - deduction

            fig_tornado = go.Figure()

            fig_tornado.add_trace(
                go.Bar(
                    y=var_names,
                    x=ltch_low - ltch_base,
                    base=ltch_base,
                    orientation="h",
                    name="Low Value",
                    marker=dict(color="#dc3545"),
                    hovertemplate="<b>%{y} (Low)</b><br>LTCH: $%{customdata[0]:.2f}<br>LCOH: $%{customdata[1]:.2f}<extra></extra>",
                    customdata=np.stack((ltch_low, lcoh_low), axis=-1),
                    xaxis="x",
                )
            )

            fig_tornado.add_trace(
                go.Bar(
                    y=var_names,
                    x=ltch_high - ltch_base,
                    base=ltch_base,
                    orientation="h",
                    name="High Value",
                    marker=dict(color="#0d6efd"),
                    hovertemplate="<b>%{y} (High)</b><br>LTCH: $%{customdata[0]:.2f}<br>LCOH: $%{customdata[1]:.2f}<extra></extra>",
                    customdata=np.stack((ltch_high, lcoh_high), axis=-1),
                    xaxis="x",
                )
            )

            mean_base = float(np.nanmean(ltch_base))
            fig_tornado.add_vline(
                x=mean_base,
                line_width=2,
                line_dash="dash",
                line_color="#333333",
                annotation_text=f"Base LTCH: ${mean_base:.2f}",
                annotation_position="top left",
            )

            min_ltch = float(min(np.nanmin(ltch_low), np.nanmin(ltch_high))) - 0.20
            max_ltch = float(max(np.nanmax(ltch_low), np.nanmax(ltch_high))) + 0.20

            fig_tornado.add_trace(
                go.Scatter(
                    x=[min_ltch - deduction, max_ltch - deduction],
                    y=[var_names[0], var_names[0]],
                    mode="markers",
                    marker=dict(opacity=0),
                    showlegend=False,
                    hoverinfo="skip",
                    xaxis="x2",
                )
            )

            fig_tornado.update_layout(
                barmode="overlay",
                height=480,
                template="plotly_white",
                xaxis=dict(
                    title=dict(text="LTCH [$ / kg H₂]", font=dict(size=22, family="Arial")),tickfont=dict(size=20, color="black"),
                    side="bottom",
                    range=[min_ltch, max_ltch],
                ),
                xaxis2=dict(
                    title=dict(text="LCOH [$ / kg H₂]", font=dict(size=22, family="Arial")),tickfont=dict(size=20, color="black"),
                    side="bottom",
                    overlaying="x",
                    anchor="free",
                    position=0.0,
                    range=[min_ltch - deduction, max_ltch - deduction],
                ),
                yaxis=dict(
                    autorange="reversed",
                    domain=[0.18, 1.0],
                    showticklabels=False,  # Labels next to bars removed
                ),
                legend=dict(orientation="h", y=1.12, x=0.7),
                margin=dict(l=50, r=50, t=50, b=40),
            )

            st.plotly_chart(fig_tornado, use_container_width=True)
        else:
            st.warning(
                "Tornado dataset is missing required LTCH low/high columns or data."
            )

    # 3. MONTE CARLO PROBABILITY HISTOGRAM
    st.markdown("")

    state_col_mc = find_column(mc_df.columns, ["State", "ST"])
    path_col_mc = find_column(mc_df.columns, ["Pathway", "Technology"])
    ltch_col_mc = find_column(
        mc_df.columns,
        ["LTCH", "LTCH ($/kg H2)", "Cost", "LTCH_$/kg_H2", "LTCH ($/kgH2)"],
    )

    filter_mc = mc_df.copy()
    if state_col_mc:
        filter_mc = filter_mc[
            filter_mc[state_col_mc].astype(str).str.upper()
            == current_state.upper()
        ]
    if path_col_mc:
        filter_mc = filter_mc[
            filter_mc[path_col_mc].astype(str).str.upper()
            == selected_pathway.upper()
        ]

    if not filter_mc.empty and ltch_col_mc:
        mc_samples = pd.to_numeric(
            filter_mc[ltch_col_mc], errors="coerce"
        ).dropna().values

        if len(mc_samples) > 0:
            p10_val = float(np.percentile(mc_samples, 10))
            p50_val = float(np.percentile(mc_samples, 50))
            p90_val = float(np.percentile(mc_samples, 90))
            mean_val = float(np.mean(mc_samples))

            fig_mc = go.Figure()

            fig_mc.add_trace(
                go.Histogram(
                    x=mc_samples,
                    nbinsx=100,
                    histnorm="probability",
                    marker_color="#0d6efd",
                    opacity=0.75,
                    name="Probability",
                )
            )

            lines = [
                (p10_val, f"P10: ${p10_val:.2f}", "#fd7e14", "dash"),
                (p50_val, f"P50: ${p50_val:.2f}", "#198754", "solid"),
                (p90_val, f"P90: ${p90_val:.2f}", "#6f42c1", "dash"),
                (mean_val, f"Mean: ${mean_val:.2f}", "#dc3545", "dot"),
            ]

            for val_line, label_line, color_line, style_line in lines:
                fig_mc.add_vline(
                    x=val_line,
                    line_width=2.5,
                    line_dash=style_line,
                    line_color=color_line,
                    annotation_text=label_line,
                    annotation_position="top right",
                )

            fig_mc.update_layout(
                height=450,
                template="plotly_white",
                xaxis=dict(
                    title=dict(
                        text="LTCH [$ / kg H₂]",
                        font=dict(size=22, family="Arial"),
                    ),tickfont=dict(size=20, color="black")
                ),
                yaxis=dict(
                    title=dict(
                        text="Probability", font=dict(size=22, family="Arial")
                    ),tickfont=dict(size=20, color="black")
                ),
                margin=dict(l=50, r=50, t=50, b=50),
            )

            st.plotly_chart(fig_mc, use_container_width=True)
        else:
            st.info("No valid numeric Monte Carlo samples found for this selection.")
    else:
        st.info("No Monte Carlo distribution data available for this selection.")


# ==========================================================
# TAB 4: VALUES TAB (INPUTS DATASET TABLES)
# ==========================================================

with tab_values:
    st.subheader("Values")

    if not inputs_df.empty:
        # Detect standard columns in InputsDataset
        var_name_col_inp = find_column(inputs_df.columns, ["Variable Name", "Variable", "Name"])
        var_unit_col_inp = find_column(inputs_df.columns, ["Variable Unit", "Unit", "Units"])
        smr_col_inp = find_column(inputs_df.columns, ["SMR"])
        smrcc_col_inp = find_column(inputs_df.columns, ["SMRCC"])
        elec_col_inp = find_column(inputs_df.columns, ["Electrolysis"])
        onsite_sm_col_inp = find_column(inputs_df.columns, ["Onsite SMR", "Onsite SM", "Onsite_SMR", "Onsite_SM"])
        onsite_elec_col_inp = find_column(inputs_df.columns, ["Onsite Electrolysis", "Onsite_Electrolysis"])

        # Create mapping dictionary for neat column display
        target_columns = [
            var_name_col_inp,
            var_unit_col_inp,
            smr_col_inp,
            smrcc_col_inp,
            elec_col_inp,
            onsite_sm_col_inp,
            onsite_elec_col_inp,
        ]
        valid_cols = [c for c in target_columns if c and c in inputs_df.columns]

        rename_map = {}
        if var_name_col_inp: rename_map[var_name_col_inp] = "Variable Name"
        if var_unit_col_inp: rename_map[var_unit_col_inp] = "Variable Unit"
        if smr_col_inp: rename_map[smr_col_inp] = "SMR"
        if smrcc_col_inp: rename_map[smrcc_col_inp] = "SMRCC"
        if elec_col_inp: rename_map[elec_col_inp] = "Electrolysis"
        if onsite_sm_col_inp: rename_map[onsite_sm_col_inp] = "Onsite SM"
        if onsite_elec_col_inp: rename_map[onsite_elec_col_inp] = "Onsite Electrolysis"

        # Function for flexible matching of target variable names
        def filter_dataset_by_vars(df, target_vars):
            if not var_name_col_inp or var_name_col_inp not in df.columns:
                return pd.DataFrame()

            def is_match(val):
                if pd.isna(val):
                    return False
                v_str = str(val).lower().replace("₂", "2").replace("subscript 2", "2").replace("_", " ").replace(",", "").strip()
                for target in target_vars:
                    t_str = str(target).lower().replace("₂", "2").replace("subscript 2", "2").replace("_", " ").replace(",", "").strip()
                    if t_str in v_str or v_str in t_str:
                        return True
                return False

            matched_df = df[df[var_name_col_inp].apply(is_match)].copy()
            if valid_cols:
                matched_df = matched_df[valid_cols].rename(columns=rename_map)
            return matched_df

        # Target variables list
        assigned_variables = [
            "CO2 Production",
            "CO₂ Production",
            "CO2 subscript 2 Production",
            "Electricity Usage",
            "Facility Capital Cost",
            "Facility Production",
            "Natural Gas Usage",
            "Tested Facility Expected Life",
            "Water usage",
            "Yearly Fixed (plant)",
            "Yearly Non-Feedstock Other Variable Costs",
            "Yearly, Non-Feedstock Other Variable Costs",
        ]

        financial_variables = [
            "Cost of Equity",
            "Cost Sensitivity",
            "Operating Costs",
            "Revenue",
            "Senior Debt Term",
            "SL Depreciation",
            "Target Main Equity Percentage",
        ]

        # 1. Assigned Values Table
        st.markdown("### Assigned Values Table")
        df_assigned = filter_dataset_by_vars(inputs_df, assigned_variables)
        if not df_assigned.empty:
            st.dataframe(df_assigned, use_container_width=True, hide_index=True)
        else:
            st.info("No matching variables found for Assigned Values Table in InputsDataset.csv.")

        st.markdown("---")

        # 2. Financial Input Values Table
        st.markdown("### Financial Input Values Table")
        df_financial = filter_dataset_by_vars(inputs_df, financial_variables)
        if not df_financial.empty:
            st.dataframe(df_financial, use_container_width=True, hide_index=True)
        else:
            st.info("No matching variables found for Financial Input Values Table in InputsDataset.csv.")

    else:
        st.warning("InputsDataset.csv not found or is empty.")


# ==========================================================
# TAB 5: EXTERNAL HTTP SITE VIEW
# ==========================================================

with tab_external:
    st.subheader("")
    st.markdown(
        ""
    )

    # Target URL for embedded site
    target_url = "https://full-tco-app-app-kd8mb5ed2bttt9rzppjkgi.streamlit.app/?embed=true"

    col_btn, col_info = st.columns([1, 4])
    with col_btn:
        st.link_button("Open Site in New Tab ↗", target_url)

    st.markdown("---")

    components.iframe(target_url, height=700, scrolling=True)
