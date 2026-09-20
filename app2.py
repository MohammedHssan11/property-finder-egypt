"""NileLens AI — Enterprise Real Estate Valuation & Investment Studio."""

from __future__ import annotations

from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from roi import FinancialAssumptions, FinancingPlan, RealEstateROIEngine
from spatial import LANDMARKS, GeospatialFeatureExtractor


st.set_page_config(
    page_title="NileLens AI | Real Estate Intelligence Studio",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&display=swap');
    html, body, [class*="css"] { font-family: 'Manrope', sans-serif; }
    .stApp {
        background:
          radial-gradient(circle at 12% 5%, rgba(26, 188, 156, .12), transparent 28%),
          radial-gradient(circle at 88% 10%, rgba(245, 183, 64, .10), transparent 26%),
          #07131f;
        color: #ecf4f7;
    }
    [data-testid="stSidebar"] { background: linear-gradient(180deg, #0a1d2d, #07131f); }
    [data-testid="stSidebar"] * { color: #e7f1f4; }
    .hero {
        padding: 1.8rem 2rem;
        border: 1px solid rgba(129, 230, 217, .20);
        border-radius: 20px;
        background: linear-gradient(120deg, rgba(11, 37, 55, .95), rgba(8, 53, 62, .78));
        box-shadow: 0 20px 60px rgba(0, 0, 0, .24);
        margin-bottom: 1.2rem;
    }
    .eyebrow { color: #f5b740; font-size: .78rem; letter-spacing: .18em; font-weight: 800; }
    .hero h1 { font-size: clamp(1.8rem, 3.5vw, 2.8rem); line-height: 1.1; margin: .45rem 0 .5rem; color: #f6fbfc; }
    .hero p { color: #afc5ce; max-width: 850px; font-size: 0.98rem; margin: 0; }
    .studio-card {
        border: 1px solid rgba(129, 230, 217, .15);
        border-radius: 16px;
        padding: 1.2rem;
        background: rgba(12, 33, 47, .78);
        margin-bottom: 1rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, .18);
    }
    .accent { color: #5eead4; }
    .gold { color: #f5b740; }
    .badge-tier {
        display: inline-block;
        padding: 0.35rem 0.8rem;
        border-radius: 8px;
        font-weight: 800;
        font-size: 0.85rem;
    }
    .tier-a { background: rgba(46, 125, 50, 0.25); color: #81c784; border: 1px solid #4caf50; }
    .tier-b { background: rgba(2, 136, 209, 0.25); color: #81d4fa; border: 1px solid #03a9f4; }
    .tier-c { background: rgba(245, 127, 23, 0.25); color: #ffe082; border: 1px solid #ffb300; }
    .tier-risk { background: rgba(198, 40, 40, 0.25); color: #ef9a9a; border: 1px solid #e53935; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_model_bundle():
    bundle_path = Path(__file__).parent / "models" / "property_price_model_bundle.joblib"
    return joblib.load(bundle_path)


bundle = load_model_bundle()
locations = bundle["location_reference"].copy()
roi_engine = RealEstateROIEngine(bundle)
geo_extractor = GeospatialFeatureExtractor(locations)

st.markdown(
    """
    <section class="hero">
      <div class="eyebrow">NILELENS AI · REAL ESTATE INTELLIGENCE SUITE</div>
      <h1>Enterprise Valuation & <span class="accent">Spatial ROI Studio</span></h1>
      <p>Multi-dimensional property analytics powered by machine learning, geodesic hub proximity models, and institutional real estate cash-flow simulators.</p>
    </section>
    """,
    unsafe_allow_html=True,
)


# Sidebar Inputs
with st.sidebar:
    st.markdown("### 🏢 Property Profile")
    category_labels = {
        "buy": "Residential · Purchase",
        "rent": "Residential · Tenancy",
        "commercial_buy": "Commercial · Purchase",
        "commercial_rent": "Commercial · Lease",
    }
    category = st.selectbox(
        "Market Category",
        list(category_labels),
        format_func=lambda value: category_labels[value],
    )

    property_types = bundle["property_types_by_category"].get(category, [])
    property_type = st.selectbox("Property Type", property_types)

    market_region_options = sorted(locations["market_region"].dropna().unique())
    market_region = st.selectbox("Region / Governorate", market_region_options)

    market_city_options = sorted(
        locations.loc[locations["market_region"].eq(market_region), "market_city"].dropna().unique()
    )
    market_city = st.selectbox("City / Zone", market_city_options)

    location_rows = locations.loc[
        locations["market_region"].eq(market_region) & locations["market_city"].eq(market_city)
    ].sort_values("listing_count", ascending=False)
    location = st.selectbox("Area / Compound", location_rows["location"].tolist())

    selected_location = location_rows.loc[location_rows["location"].eq(location)].iloc[0]

    st.divider()
    size = st.slider("Built-up Size (m²)", 15, 1500, 160, 5)
    r1, r2 = st.columns(2)
    with r1:
        bedrooms = st.number_input("Bedrooms", min_value=0, max_value=15, value=3)
    with r2:
        bathrooms = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)

    amenities_count = st.slider("Amenities Count", 0, 30, 6)
    images_count = st.slider("Listing Images", 0, 30, 4)

    st.divider()
    st.markdown("### 💳 Financing Assumptions")
    fin_type = st.selectbox("Financing Structure", ["developer_installment", "mortgage", "cash"])
    down_pct = st.slider("Down Payment %", 5, 50, 15, 5) / 100.0
    tenure_yrs = st.slider("Tenure (Years)", 1, 15, 7)
    interest_rate = 0.0
    if fin_type == "mortgage":
        interest_rate = st.slider("Mortgage Rate %", 10.0, 30.0, 19.5, 0.5) / 100.0


# Execute Valuation & Analytics
is_commercial = category.startswith("commercial")
bedroom_value = np.nan if is_commercial and bedrooms == 0 else float(bedrooms)
bathroom_value = np.nan if is_commercial and bathrooms == 0 else float(bathrooms)

latitude = selected_location["latitude"]
longitude = selected_location["longitude"]
coord_missing = int(pd.isna(latitude) or pd.isna(longitude))

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
    "coordinate_missing": coord_missing,
    "category": category,
    "price_period": "monthly" if "rent" in category else "sell",
    "property_type": property_type,
    "market_region": market_region,
    "market_city": market_city,
    "local_area": selected_location["local_area"],
    "location": location,
}])

pred_log = bundle["model"].predict(input_row[bundle["feature_columns"]])[0]
predicted_price = max(1000.0, float(np.expm1(pred_log)))
price_sqm = predicted_price / max(1.0, size)

geo_profile = geo_extractor.extract_features(
    latitude=latitude,
    longitude=longitude,
    location_text=location,
    market_region=market_region,
    market_city=market_city,
)

# Dual Valuation for ROI
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
    "amenities_count": amenities_count,
    "images_count": images_count,
    "latitude": latitude,
    "longitude": longitude,
}
buy_val, rent_val = roi_engine.estimate_dual_valuation(base_dict)
if "buy" in category:
    buy_val = predicted_price
else:
    rent_val = predicted_price

financing_plan = FinancingPlan(
    plan_type=fin_type,
    down_payment_pct=down_pct,
    duration_years=tenure_yrs,
    annual_interest_rate=interest_rate,
)
roi_report = roi_engine.evaluate_investment(
    purchase_price_egp=buy_val,
    monthly_rent_egp=rent_val,
    financing=financing_plan,
    accessibility_score=geo_profile.accessibility_score,
    is_compound=geo_profile.is_compound,
)


tabs = st.tabs([
    "📍 Valuation & Spatial Intelligence",
    "📈 Investment ROI & Financial Simulator",
    "🗺️ Micro-Market Spatial Heatmap",
    "⚖️ Portfolio Asset Comparator",
])


# =========================================================================
# TAB 1: Valuation & Spatial Intelligence
# =========================================================================
with tabs[0]:
    col_v1, col_v2, col_v3 = st.columns(3)
    suffix = " EGP / mo" if "rent" in category else " EGP"
    with col_v1:
        st.metric("Estimated Fair Value", f"{predicted_price:,.0f}{suffix}")
    with col_v2:
        unit_suffix = " EGP/m²/mo" if "rent" in category else " EGP/m²"
        st.metric("Price per Square Meter", f"{price_sqm:,.0f}{unit_suffix}")
    with col_v3:
        st.metric("Accessibility Index", f"{geo_profile.accessibility_score:.1f} / 100")

    st.markdown("---")
    col_map, col_telemetry = st.columns([3, 2])

    with col_telemetry:
        st.markdown("#### 🎯 Spatial Landmark Proximity")
        st.markdown(f"**Nearest Business Hub:** `{geo_profile.nearest_hub.replace('_', ' ').title()}` ({geo_profile.nearest_hub_distance_km:.1f} km)")
        st.markdown(f"**Community Classification:** `{geo_profile.compound_tier}`")
        if geo_profile.matched_compound:
            st.caption(f"Matched Community Tag: *{geo_profile.matched_compound}*")

        dist_data = [
            {"Landmark": k.replace("_", " ").title(), "Distance (km)": v}
            for k, v in geo_profile.landmark_distances_km.items()
        ]
        df_dist = pd.DataFrame(dist_data).sort_values("Distance (km)")
        st.dataframe(df_dist, use_container_width=True, hide_index=True)

    with col_map:
        st.markdown("#### 🌍 Egyptian Landmark Network Map")
        map_points = [
            {"Name": f"Property: {location}", "lat": geo_profile.latitude, "lon": geo_profile.longitude, "Type": "Subject Property", "Color": "#1abc9c"}
        ]
        for name, (l_lat, l_lon) in LANDMARKS.items():
            map_points.append({
                "Name": name.replace("_", " ").title(),
                "lat": l_lat,
                "lon": l_lon,
                "Type": "Major Commercial Hub",
                "Color": "#f5b740"
            })
        df_map = pd.DataFrame(map_points)
        fig_map = px.scatter_mapbox(
            df_map,
            lat="lat",
            lon="lon",
            text="Name",
            color="Type",
            size=[14 if t == "Subject Property" else 10 for t in df_map["Type"]],
            color_discrete_map={"Subject Property": "#5eead4", "Major Commercial Hub": "#f5b740"},
            zoom=7.5,
            center={"lat": geo_profile.latitude, "lon": geo_profile.longitude},
            mapbox_style="carto-darkmatter",
            height=380,
        )
        fig_map.update_layout(margin=dict(l=0, r=0, t=0, b=0), legend=dict(x=0.02, y=0.98))
        st.plotly_chart(fig_map, use_container_width=True)


# =========================================================================
# TAB 2: Investment ROI & Financial Simulator
# =========================================================================
with tabs[1]:
    st.markdown("### 📊 Comprehensive Real Estate Financial Model")
    st.caption("Dual Buy-vs-Rent model evaluating rental yields, installment cash flow, and inflation capital appreciation.")

    r1, r2, r3, r4 = st.columns(4)
    r1.metric("Purchase Price", f"{roi_report.purchase_price_egp:,.0f} EGP")
    r2.metric("Monthly Rent (Yield Base)", f"{roi_report.monthly_rent_egp:,.0f} EGP/mo")
    r3.metric("Gross Rental Yield", f"{roi_report.gross_rental_yield_pct:.2f}%")
    r4.metric("Net Rental Yield (NOI)", f"{roi_report.net_rental_yield_pct:.2f}%")

    st.markdown("---")
    c_fin1, c_fin2 = st.columns(2)

    with c_fin1:
        st.markdown(f"#### 💳 Financing & Cash Flow (`{fin_type.upper()}`)")
        f_df = pd.DataFrame([
            {"Metric": "Initial Outlay (Down Payment + Closing)", "Value": f"{roi_report.initial_equity_invested_egp:,.0f} EGP"},
            {"Metric": "Monthly Debt / Installment Payment", "Value": f"{roi_report.monthly_installment_egp:,.0f} EGP/mo"},
            {"Metric": "Annual Debt Service", "Value": f"{roi_report.annual_debt_service_egp:,.0f} EGP/yr"},
            {"Metric": "Net Operating Income (NOI)", "Value": f"{roi_report.net_operating_income_egp:,.0f} EGP/yr"},
            {"Metric": "Annual Net Cash Flow (Pre-Tax)", "Value": f"{roi_report.annual_net_cash_flow_egp:,.0f} EGP/yr"},
            {"Metric": "Cash-on-Cash Return", "Value": f"{roi_report.cash_on_cash_return_pct:.2f}%"},
            {"Metric": "Payback Horizon", "Value": f"{roi_report.payback_period_years} Years"},
        ])
        st.table(f_df)

    with c_fin2:
        st.markdown("#### 🏆 Investment Scorecard")
        score = roi_report.investment_score
        tier_class = "tier-a" if "A" in roi_report.investment_tier else ("tier-b" if "B" in roi_report.investment_tier else "tier-c")
        st.markdown(
            f"""
            <div class="studio-card">
              <div style="font-size: 0.85rem; color: #8fb0bc;">COMPOSITE OPPORTUNITY INDEX</div>
              <div style="font-size: 2.8rem; font-weight: 800; color: #5eead4;">{score:.1f} <span style="font-size: 1.2rem; color: #8fb0bc;">/ 100</span></div>
              <div style="margin-top: 0.5rem;"><span class="badge-tier {tier_class}">{roi_report.investment_tier}</span></div>
              <p style="margin-top: 0.8rem; font-size: 0.88rem; color: #afc5ce;">
                Integrates Gross/Net Yields, regional accessibility index, cash-flow coverage, and gated development premium into an institutional rating.
              </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📈 Multi-Scenario Capital Appreciation Forecast")
    years_range = list(range(1, 11))
    c_series = [roi_report.purchase_price_egp * ((1.12) ** y) for y in years_range]
    b_series = [roi_report.purchase_price_egp * ((1.18) ** y) for y in years_range]
    h_series = [roi_report.purchase_price_egp * ((1.25) ** y) for y in years_range]

    fig_app = go.Figure()
    fig_app.add_trace(go.Scatter(x=years_range, y=c_series, mode="lines+markers", name="Conservative (12% CAGR)", line=dict(color="#81c784", dash="dot")))
    fig_app.add_trace(go.Scatter(x=years_range, y=b_series, mode="lines+markers", name="Base Case (18% CAGR)", line=dict(color="#5eead4", width=3)))
    fig_app.add_trace(go.Scatter(x=years_range, y=h_series, mode="lines+markers", name="High Inflation Hedge (25% CAGR)", line=dict(color="#f5b740", dash="dash")))

    fig_app.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(12, 33, 47, .78)",
        xaxis=dict(title="Holding Period (Years)", tickmode="linear", dtick=1),
        yaxis=dict(title="Nominal Value (EGP)", tickformat=",.0f"),
        height=360,
        margin=dict(l=20, r=20, t=30, b=20),
    )
    st.plotly_chart(fig_app, use_container_width=True)


# =========================================================================
# TAB 3: Spatial Market Heatmap & Micro-Markets
# =========================================================================
with tabs[2]:
    st.markdown("### 🗺️ Micro-Market Spatial Explorer")
    st.caption("Distribution of 2,900+ Egyptian real estate micro-markets across price tiers, densities, and regions.")

    sub_locs = locations.copy()
    sub_locs = sub_locs.dropna(subset=["latitude", "longitude", "median_price"])

    col_f1, col_f2 = st.columns(2)
    with col_f1:
        sel_reg = st.multiselect("Filter Regions", sorted(sub_locs["market_region"].unique()), default=["Cairo", "Giza"])
    with col_f2:
        max_price_filter = st.slider("Max Median Price (Millions EGP)", 1.0, 50.0, 30.0, 1.0) * 1_000_000

    if sel_reg:
        sub_locs = sub_locs[sub_locs["market_region"].isin(sel_reg)]
    sub_locs = sub_locs[sub_locs["median_price"] <= max_price_filter]

    fig_scat = px.scatter_mapbox(
        sub_locs.head(400),
        lat="latitude",
        lon="longitude",
        hover_name="location",
        hover_data={"median_price": ":,.0f", "listing_count": True, "market_city": True},
        color="median_price",
        size="listing_count",
        color_continuous_scale="Viridis",
        zoom=9.0,
        mapbox_style="carto-darkmatter",
        height=520,
    )
    fig_scat.update_layout(margin=dict(l=0, r=0, t=0, b=0))
    st.plotly_chart(fig_scat, use_container_width=True)


# =========================================================================
# TAB 4: Portfolio Investment Comparator
# =========================================================================
with tabs[3]:
    st.markdown("### ⚖️ Side-by-Side Asset Comparator")
    st.caption("Evaluate multiple real estate investment opportunities to identify optimal yield and risk profiles.")

    comp_options = [
        {"Name": f"Option A: {location} ({property_type})", "size": size, "beds": bedrooms, "baths": bathrooms, "loc": location, "city": market_city, "reg": market_region, "ptype": property_type},
        {"Name": "Option B: Tagamoa Golden Square (Apartment)", "size": 180.0, "beds": 3, "baths": 3, "loc": "Mivida", "city": "New Cairo - El Tagamoa", "reg": "Cairo", "ptype": "Apartment"},
        {"Name": "Option C: Sheikh Zayed Central (Villa)", "size": 250.0, "beds": 4, "baths": 4, "loc": "Palm Hills", "city": "Sheikh Zayed", "reg": "Giza", "ptype": "Villa"},
    ]

    comp_results = []
    for opt in comp_options:
        loc_r = locations.loc[locations["location"].str.contains(opt["loc"], case=False, na=False)]
        loc_row_c = loc_r.iloc[0] if not loc_r.empty else selected_location
        b_input = {
            "size": opt["size"],
            "bedrooms": opt["beds"],
            "bathrooms": opt["baths"],
            "property_type": opt["ptype"],
            "market_region": opt["reg"],
            "market_city": opt["city"],
            "local_area": loc_row_c["local_area"],
            "location": loc_row_c["location"],
            "latitude": loc_row_c["latitude"],
            "longitude": loc_row_c["longitude"],
        }
        b_val, r_val = roi_engine.estimate_dual_valuation(b_input)
        g_prof = geo_extractor.extract_features(loc_row_c["latitude"], loc_row_c["longitude"], location_text=opt["loc"])
        r_rep = roi_engine.evaluate_investment(b_val, r_val, accessibility_score=g_prof.accessibility_score, is_compound=g_prof.is_compound)
        comp_results.append({
            "Asset": opt["Name"],
            "Size (m²)": opt["size"],
            "Purchase Price (EGP)": f"{r_rep.purchase_price_egp:,.0f}",
            "Monthly Rent (EGP)": f"{r_rep.monthly_rent_egp:,.0f}",
            "Gross Yield %": f"{r_rep.gross_rental_yield_pct:.2f}%",
            "Net Yield %": f"{r_rep.net_rental_yield_pct:.2f}%",
            "Accessibility": f"{g_prof.accessibility_score:.1f}",
            "Investment Score": f"{r_rep.investment_score:.1f}",
            "Tier": r_rep.investment_tier,
        })

    st.table(pd.DataFrame(comp_results))
