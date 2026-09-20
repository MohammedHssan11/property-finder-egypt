from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from roi import RealEstateROIEngine
from spatial import GeospatialFeatureExtractor


st.set_page_config(
    page_title="Egypt Property Price & ROI Estimator",
    page_icon="🏠",
    layout="centered",
)


@st.cache_resource
def load_model_bundle():
    bundle_path = Path(__file__).parent / "models" / "property_price_model_bundle.joblib"
    return joblib.load(bundle_path)


bundle = load_model_bundle()
locations = bundle["location_reference"].copy()
roi_engine = RealEstateROIEngine(bundle)
geo_extractor = GeospatialFeatureExtractor(locations)

st.title("🏠 Egypt Property Price & ROI Estimator")
st.write("Machine learning price intelligence and investment yield estimator powered by Property Finder Egypt market listings.")

with st.expander("ℹ️ Market & Modeling Context", expanded=False):
    st.markdown(
        """
        - **Valuation Engine**: Scikit-Learn Pipeline with Random Forest Regressor trained on 62,000+ real Egypt listings.
        - **Geospatial Features**: Automatic geodesic landmark proximity (Cairo CBD, New Capital, Sheikh Zayed, Golden Square, Airport).
        - **Investment Analytics**: Dual-model Buy vs. Rent yield estimation, Net Operating Income (NOI), and capital appreciation.
        """
    )

category = st.selectbox(
    "Market category",
    ["buy", "rent", "commercial_buy", "commercial_rent"],
    help="Select buy for asset acquisition or rent for tenancy valuation.",
)

property_types = bundle["property_types_by_category"].get(category, [])
property_type = st.selectbox("Property type", property_types)

col_reg, col_city = st.columns(2)
with col_reg:
    market_region = st.selectbox(
        "Market region",
        sorted(locations["market_region"].dropna().unique()),
    )

with col_city:
    market_city_options = sorted(
        locations.loc[
            locations["market_region"].eq(market_region),
            "market_city",
        ].dropna().unique()
    )
    market_city = st.selectbox("Market city / zone", market_city_options)

location_options = (
    locations.loc[
        locations["market_region"].eq(market_region)
        & locations["market_city"].eq(market_city)
    ]
    .sort_values("listing_count", ascending=False)["location"]
    .tolist()
)
location = st.selectbox("Specific location / compound", location_options)

selected_location = locations.loc[locations["location"].eq(location)].iloc[0]

col_size, col_beds, col_baths = st.columns(3)
with col_size:
    size = st.number_input("Built-up Size (m²)", min_value=10.0, max_value=100000.0, value=150.0, step=10.0)
with col_beds:
    bedrooms = st.number_input("Bedrooms", min_value=0, max_value=20, value=3)
with col_baths:
    bathrooms = st.number_input("Bathrooms", min_value=0, max_value=20, value=2)

col_am, col_img = st.columns(2)
with col_am:
    amenities_count = st.number_input("Amenities count", min_value=0, max_value=100, value=5)
with col_img:
    images_count = st.number_input("Listing images count", min_value=0, max_value=100, value=2)

if st.button("Generate Valuation & ROI Intelligence", type="primary", use_container_width=True):
    is_commercial = category.startswith("commercial")
    bedroom_value = np.nan if is_commercial and bedrooms == 0 else float(bedrooms)
    bathroom_value = np.nan if is_commercial and bathrooms == 0 else float(bathrooms)

    latitude = selected_location["latitude"]
    longitude = selected_location["longitude"]
    coordinate_missing = int(pd.isna(latitude) or pd.isna(longitude))

    input_row = pd.DataFrame([{
        "bedrooms": bedroom_value,
        "bathrooms": bathroom_value,
        "size": float(size),
        "latitude": latitude,
        "longitude": longitude,
        "amenities_count": int(amenities_count),
        "images_count": int(images_count),
        "total_rooms": np.nansum([bedroom_value, bathroom_value]),
        "log_size": np.log1p(size),
        "bedroom_missing": int(pd.isna(bedroom_value)),
        "bathroom_missing": int(pd.isna(bathroom_value)),
        "coordinate_missing": coordinate_missing,
        "category": category,
        "price_period": "monthly" if "rent" in category else "sell",
        "property_type": property_type,
        "market_region": market_region,
        "market_city": market_city,
        "local_area": selected_location["local_area"],
        "location": location,
    }])

    predicted_price = np.expm1(
        bundle["model"].predict(input_row[bundle["feature_columns"]])[0]
    )
    predicted_price = max(1000.0, predicted_price)
    price_per_sqm = predicted_price / max(1.0, size)

    cat_summary = bundle.get("category_summary", {}).get(category, {})
    ape = cat_summary.get("median_APE", 0.16)
    low_price = predicted_price * (1.0 - ape)
    high_price = predicted_price * (1.0 + ape)

    # Geospatial profile
    geo_prof = geo_extractor.extract_features(
        latitude=latitude,
        longitude=longitude,
        location_text=location,
        market_region=market_region,
        market_city=market_city,
    )

    st.markdown("---")
    suffix = " EGP / month" if "rent" in category else " EGP"

    st.subheader("💰 Valuation Summary")
    c1, c2 = st.columns(2)
    with c1:
        st.metric("Estimated Asking Price", f"{predicted_price:,.0f}{suffix}")
    with c2:
        unit_suffix = "/m²/mo" if "rent" in category else "/m²"
        st.metric("Price per Square Meter", f"{price_per_sqm:,.0f} EGP{unit_suffix}")

    st.caption(f"Confidence Range (±{ape:.0%}): **{low_price:,.0f} — {high_price:,.0f} EGP**")

    # Spatial Proximity Card
    st.markdown("---")
    st.subheader("📍 Geospatial & Proximity Intelligence")
    g1, g2, g3 = st.columns(3)
    g1.metric("Nearest Hub", f"{geo_prof.nearest_hub_distance_km:.1f} km", geo_prof.nearest_hub.replace("_", " ").title())
    g2.metric("Cairo CBD", f"{geo_prof.landmark_distances_km.get('cairo_cbd', 0.0):.1f} km")
    g3.metric("New Capital", f"{geo_prof.landmark_distances_km.get('new_capital_financial', 0.0):.1f} km")

    g4, g5 = st.columns(2)
    with g4:
        st.metric("Accessibility Index", f"{geo_prof.accessibility_score:.1f} / 100")
    with g5:
        st.metric("Community Status", geo_prof.compound_tier)

    # Real Estate ROI Preview (if buy property)
    if "rent" not in category:
        st.markdown("---")
        st.subheader("📈 Real Estate Investment & ROI Preview")
        # Dual valuation for rent
        base_dict = {
            "size": size,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "property_type": property_type,
            "category": category,
            "market_region": market_region,
            "market_city": market_city,
            "local_area": selected_location["local_area"],
            "location": location,
            "latitude": latitude,
            "longitude": longitude,
            "amenities_count": amenities_count,
            "images_count": images_count,
        }
        buy_val, rent_val = roi_engine.estimate_dual_valuation(base_dict)
        roi_report = roi_engine.evaluate_investment(
            purchase_price_egp=predicted_price,
            monthly_rent_egp=rent_val,
            accessibility_score=geo_prof.accessibility_score,
            is_compound=geo_prof.is_compound,
        )

        r1, r2, r3 = st.columns(3)
        r1.metric("Gross Rental Yield", f"{roi_report.gross_rental_yield_pct:.2f}%")
        r2.metric("Net Rental Yield", f"{roi_report.net_rental_yield_pct:.2f}%")
        r3.metric("Investment Tier", roi_report.investment_tier)

        st.caption(
            f"Expected Monthly Rent: **{roi_report.monthly_rent_egp:,.0f} EGP/month** | "
            f"5-Year Base Appreciation (18% CAGR): **{roi_report.appreciation_5yr['base_case']:,.0f} EGP**"
        )
