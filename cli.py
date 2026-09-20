"""
Enterprise Command-Line Interface for Property Finder Egypt.

Provides subcommands for price estimation, geospatial auditing, investment ROI analysis,
and batch portfolio ranking.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd

from roi import FinancialAssumptions, FinancingPlan, RealEstateROIEngine
from spatial import GeospatialFeatureExtractor, haversine_distance


DEFAULT_MODEL_PATH = Path(__file__).resolve().parent / "models" / "property_price_model_bundle.joblib"


def load_bundle(model_path: Path = DEFAULT_MODEL_PATH) -> Dict[str, Any]:
    """Loads the production model bundle."""
    if not model_path.exists():
        raise FileNotFoundError(f"Model bundle not found at: {model_path}")
    return joblib.load(model_path)


def resolve_location(
    locations_df: pd.DataFrame,
    location_name: Optional[str] = None,
    region: Optional[str] = None,
    city: Optional[str] = None,
) -> pd.Series:
    """Finds best matching location row from reference database."""
    sub = locations_df
    if region:
        reg_match = sub[sub["market_region"].str.lower() == region.lower()]
        if not reg_match.empty:
            sub = reg_match
    if city:
        city_match = sub[sub["market_city"].str.lower() == city.lower()]
        if not city_match.empty:
            sub = city_match
    if location_name:
        loc_match = sub[sub["location"].str.lower().str.contains(location_name.lower(), regex=False)]
        if not loc_match.empty:
            return loc_match.sort_values("listing_count", ascending=False).iloc[0]

    if not sub.empty:
        return sub.sort_values("listing_count", ascending=False).iloc[0]
    return locations_df.iloc[0]


def build_feature_row(
    bundle: Dict[str, Any],
    size: float,
    bedrooms: int,
    bathrooms: int,
    category: str,
    property_type: str,
    loc_row: pd.Series,
    amenities: int = 5,
    images: int = 2,
    lat_override: Optional[float] = None,
    lon_override: Optional[float] = None,
) -> pd.DataFrame:
    """Constructs single-row DataFrame aligned with trained model features."""
    is_commercial = category.startswith("commercial")
    bed_val = np.nan if is_commercial and bedrooms == 0 else float(bedrooms)
    bath_val = np.nan if is_commercial and bathrooms == 0 else float(bathrooms)

    lat = lat_override if lat_override is not None else loc_row.get("latitude")
    lon = lon_override if lon_override is not None else loc_row.get("longitude")
    coord_missing = int(pd.isna(lat) or pd.isna(lon))

    row = {
        "bedrooms": bed_val,
        "bathrooms": bath_val,
        "size": float(size),
        "latitude": lat,
        "longitude": lon,
        "amenities_count": int(amenities),
        "images_count": int(images),
        "total_rooms": np.nansum([bed_val, bath_val]),
        "log_size": np.log1p(size),
        "bedroom_missing": int(pd.isna(bed_val)),
        "bathroom_missing": int(pd.isna(bath_val)),
        "coordinate_missing": coord_missing,
        "category": category,
        "price_period": "monthly" if "rent" in category else "sell",
        "property_type": property_type,
        "market_region": loc_row.get("market_region", "Cairo"),
        "market_city": loc_row.get("market_city", "New Cairo - El Tagamoa"),
        "local_area": loc_row.get("local_area", ""),
        "location": loc_row.get("location", ""),
    }

    feature_cols = bundle["feature_columns"]
    df = pd.DataFrame([row])
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_cols]


def cmd_predict(args: argparse.Namespace) -> None:
    bundle = load_bundle(Path(args.model_bundle))
    loc_row = resolve_location(
        bundle["location_reference"],
        location_name=args.location,
        region=args.market_region,
        city=args.market_city,
    )

    input_df = build_feature_row(
        bundle=bundle,
        size=args.size,
        bedrooms=args.bedrooms,
        bathrooms=args.bathrooms,
        category=args.category,
        property_type=args.property_type,
        loc_row=loc_row,
        amenities=args.amenities,
        images=args.images,
        lat_override=args.lat,
        lon_override=args.lon,
    )

    pred_log = bundle["model"].predict(input_df)[0]
    pred_price = float(np.expm1(pred_log))
    price_per_sqm = pred_price / max(1.0, args.size)

    # Error margin estimation based on holdout category median APE
    cat_summary = bundle.get("category_summary", {}).get(args.category, {})
    ape = cat_summary.get("median_APE", 0.16)
    low_price = pred_price * (1.0 - ape)
    high_price = pred_price * (1.0 + ape)

    # Geospatial profile
    extractor = GeospatialFeatureExtractor(bundle.get("location_reference"))
    geo_prof = extractor.extract_features(
        latitude=input_df["latitude"].iloc[0],
        longitude=input_df["longitude"].iloc[0],
        location_text=loc_row["location"],
        market_region=loc_row["market_region"],
        market_city=loc_row["market_city"],
    )

    if args.json:
        out = {
            "predicted_price_egp": round(pred_price, 2),
            "price_per_sqm_egp": round(price_per_sqm, 2),
            "confidence_lower_egp": round(low_price, 2),
            "confidence_upper_egp": round(high_price, 2),
            "category": args.category,
            "property_type": args.property_type,
            "size_sqm": args.size,
            "location": loc_row["location"],
            "market_city": loc_row["market_city"],
            "geospatial": extractor.to_dict(geo_prof),
        }
        print(json.dumps(out, indent=2))
        return

    period_str = "/month" if "rent" in args.category else " (Total Asking)"
    print("=" * 65)
    print("           PROPERTY FINDER EGYPT — PRICE ESTIMATION")
    print("=" * 65)
    print(f"Location      : {loc_row['location']} ({loc_row['market_city']}, {loc_row['market_region']})")
    print(f"Property Type : {args.property_type} ({args.category}) | Size: {args.size:.0f} m²")
    print(f"Specs         : {args.bedrooms} Beds, {args.bathrooms} Baths | Amenities: {args.amenities}")
    print("-" * 65)
    print(f"ESTIMATED VALUATION : {pred_price:,.0f} EGP{period_str}")
    print(f"Price per m²        : {price_per_sqm:,.0f} EGP/m²")
    print(f"Expected Range      : {low_price:,.0f} - {high_price:,.0f} EGP (±{ape:.0%})")
    print("-" * 65)
    print("SPATIAL & PROXIMITY TELEMETRY:")
    print(f"  • Nearest Major Hub  : {geo_prof.nearest_hub} ({geo_prof.nearest_hub_distance_km:.1f} km)")
    print(f"  • Cairo CBD Distance : {geo_prof.landmark_distances_km.get('cairo_cbd', 0.0):.1f} km")
    print(f"  • New Capital Dist.  : {geo_prof.landmark_distances_km.get('new_capital_financial', 0.0):.1f} km")
    print(f"  • Airport Distance   : {geo_prof.landmark_distances_km.get('cairo_airport', 0.0):.1f} km")
    print(f"  • Accessibility Index: {geo_prof.accessibility_score:.1f} / 100")
    print(f"  • Community Status   : {geo_prof.compound_tier}")
    print("=" * 65)


def cmd_roi(args: argparse.Namespace) -> None:
    bundle = load_bundle(Path(args.model_bundle))
    loc_row = resolve_location(
        bundle["location_reference"],
        location_name=args.location,
        region=args.market_region,
        city=args.market_city,
    )

    roi_engine = RealEstateROIEngine(bundle)
    extractor = GeospatialFeatureExtractor(bundle.get("location_reference"))

    lat = args.lat if args.lat is not None else loc_row.get("latitude")
    lon = args.lon if args.lon is not None else loc_row.get("longitude")

    base_input = {
        "size": args.size,
        "bedrooms": args.bedrooms,
        "bathrooms": args.bathrooms,
        "property_type": args.property_type,
        "market_region": loc_row["market_region"],
        "market_city": loc_row["market_city"],
        "local_area": loc_row["local_area"],
        "location": loc_row["location"],
        "amenities_count": args.amenities,
        "images_count": args.images,
        "latitude": lat,
        "longitude": lon,
        "coordinate_missing": int(pd.isna(lat) or pd.isna(lon)),
    }

    # Dual valuation: Buy & Rent
    if args.purchase_price and args.monthly_rent:
        buy_price = float(args.purchase_price)
        monthly_rent = float(args.monthly_rent)
    else:
        buy_price, monthly_rent = roi_engine.estimate_dual_valuation(base_input)
        if args.purchase_price:
            buy_price = float(args.purchase_price)
        if args.monthly_rent:
            monthly_rent = float(args.monthly_rent)

    geo_prof = extractor.extract_features(
        latitude=lat,
        longitude=lon,
        location_text=loc_row["location"],
        market_region=loc_row["market_region"],
        market_city=loc_row["market_city"],
    )

    financing = FinancingPlan(
        plan_type=args.financing,
        down_payment_pct=args.down_payment,
        duration_years=args.years,
        annual_interest_rate=args.interest_rate,
    )

    report = roi_engine.evaluate_investment(
        purchase_price_egp=buy_price,
        monthly_rent_egp=monthly_rent,
        financing=financing,
        accessibility_score=geo_prof.accessibility_score,
        is_compound=geo_prof.is_compound,
    )

    if args.json:
        out = {
            "purchase_price_egp": report.purchase_price_egp,
            "monthly_rent_egp": report.monthly_rent_egp,
            "annual_gross_rent_egp": report.annual_gross_rent_egp,
            "net_operating_income_egp": report.net_operating_income_egp,
            "gross_rental_yield_pct": report.gross_rental_yield_pct,
            "net_rental_yield_pct": report.net_rental_yield_pct,
            "cash_on_cash_return_pct": report.cash_on_cash_return_pct,
            "payback_period_years": report.payback_period_years,
            "investment_score": report.investment_score,
            "investment_tier": report.investment_tier,
            "appreciation_5yr": report.appreciation_5yr,
            "financing": {
                "plan": financing.plan_type,
                "initial_equity_egp": report.initial_equity_invested_egp,
                "monthly_installment_egp": report.monthly_installment_egp,
                "annual_net_cash_flow_egp": report.annual_net_cash_flow_egp,
            },
        }
        print(json.dumps(out, indent=2))
        return

    print("=" * 70)
    print("          PROPERTY FINDER EGYPT — REAL ESTATE ROI AUDIT")
    print("=" * 70)
    print(f"Asset Location    : {loc_row['location']} ({loc_row['market_city']})")
    print(f"Purchase Price    : {report.purchase_price_egp:,.0f} EGP")
    print(f"Estimated Rent    : {report.monthly_rent_egp:,.0f} EGP/month ({report.annual_gross_rent_egp:,.0f} EGP/year)")
    print("-" * 70)
    print("YIELD & PERFORMANCE METRICS:")
    print(f"  • Gross Rental Yield     : {report.gross_rental_yield_pct:.2f}%")
    print(f"  • Net Rental Yield (NOI) : {report.net_rental_yield_pct:.2f}% ({report.net_operating_income_egp:,.0f} EGP/yr)")
    print(f"  • Operating Expenses     : {report.operating_expenses_egp:,.0f} EGP/year")
    print(f"  • Payback Period         : {report.payback_period_years} Years")
    print(f"  • Investment Score       : {report.investment_score:.1f} / 100 [{report.investment_tier}]")
    print("-" * 70)
    print(f"FINANCING CASH FLOW ({financing.plan_type.upper()}):")
    print(f"  • Initial Invested Equity: {report.initial_equity_invested_egp:,.0f} EGP ({financing.down_payment_pct:.0%} down + closing)")
    print(f"  • Monthly Installment    : {report.monthly_installment_egp:,.0f} EGP/month ({financing.duration_years} Years)")
    print(f"  • Annual Net Cash Flow   : {report.annual_net_cash_flow_egp:,.0f} EGP/year")
    print(f"  • Cash-on-Cash Return    : {report.cash_on_cash_return_pct:.2f}%")
    print("-" * 70)
    print("5-YEAR CAPITAL APPRECIATION PROJECTIONS:")
    print(f"  • Conservative (12% CAGR): {report.appreciation_5yr['conservative']:,.0f} EGP")
    print(f"  • Base Case (18% CAGR)   : {report.appreciation_5yr['base_case']:,.0f} EGP")
    print(f"  • High Inflation (25%)   : {report.appreciation_5yr['high_growth']:,.0f} EGP")
    print("=" * 70)


def cmd_spatial_audit(args: argparse.Namespace) -> None:
    bundle = load_bundle(Path(args.model_bundle))
    loc_row = resolve_location(
        bundle["location_reference"],
        location_name=args.location,
        region=args.market_region,
        city=args.market_city,
    )

    extractor = GeospatialFeatureExtractor(bundle.get("location_reference"))
    lat = args.lat if args.lat is not None else loc_row.get("latitude")
    lon = args.lon if args.lon is not None else loc_row.get("longitude")

    prof = extractor.extract_features(
        latitude=lat,
        longitude=lon,
        location_text=loc_row["location"],
        market_region=loc_row["market_region"],
        market_city=loc_row["market_city"],
    )

    if args.json:
        print(json.dumps(extractor.to_dict(prof), indent=2))
        return

    print("=" * 65)
    print("          GEOSPATIAL SPATIAL AUDIT — PROPERTY FINDER")
    print("=" * 65)
    print(f"Location Target  : {loc_row['location']}")
    print(f"Coordinates      : Lat {prof.latitude:.4f}, Lon {prof.longitude:.4f} (Imputed: {prof.coordinate_imputed})")
    print(f"Development Type : {prof.compound_tier}")
    print(f"Accessibility    : {prof.accessibility_score:.1f} / 100")
    print(f"Active Listings  : {prof.local_density_count} in reference database")
    print("-" * 65)
    print("LANDMARK DISTANCES:")
    for hub, dist in prof.landmark_distances_km.items():
        is_nearest = " (NEAREST)" if hub == prof.nearest_hub else ""
        print(f"  • {hub.replace('_', ' ').title():<28} : {dist:>6.1f} km{is_nearest}")
    print("=" * 65)


def cmd_batch(args: argparse.Namespace) -> None:
    bundle = load_bundle(Path(args.model_bundle))
    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Batch input file not found: {in_path}")

    if in_path.suffix.lower() == ".json":
        records = pd.read_json(in_path)
    else:
        records = pd.read_csv(in_path)

    print(f"Evaluating {len(records)} listings from {in_path}...")
    roi_engine = RealEstateROIEngine(bundle)
    extractor = GeospatialFeatureExtractor(bundle.get("location_reference"))

    results = []
    for _, item in records.iterrows():
        size = float(item.get("size", 150.0))
        beds = int(item.get("bedrooms", 3))
        baths = int(item.get("bathrooms", 2))
        loc_name = str(item.get("location", "New Cairo"))
        cat = str(item.get("category", "buy"))
        ptype = str(item.get("property_type", "Apartment"))

        loc_row = resolve_location(bundle["location_reference"], location_name=loc_name)
        lat = item.get("latitude", loc_row.get("latitude"))
        lon = item.get("longitude", loc_row.get("longitude"))

        # Predict Buy & Rent
        base_input = {
            "size": size,
            "bedrooms": beds,
            "bathrooms": baths,
            "property_type": ptype,
            "market_region": loc_row["market_region"],
            "market_city": loc_row["market_city"],
            "local_area": loc_row["local_area"],
            "location": loc_row["location"],
            "amenities_count": item.get("amenities_count", 5),
            "images_count": item.get("images_count", 2),
            "latitude": lat,
            "longitude": lon,
            "coordinate_missing": int(pd.isna(lat) or pd.isna(lon)),
        }

        buy_p, rent_p = roi_engine.estimate_dual_valuation(base_input)
        geo_prof = extractor.extract_features(lat, lon, location_text=loc_row["location"])
        roi_rep = roi_engine.evaluate_investment(
            buy_p, rent_p,
            accessibility_score=geo_prof.accessibility_score,
            is_compound=geo_prof.is_compound,
        )

        results.append({
            "location": loc_row["location"],
            "size": size,
            "bedrooms": beds,
            "purchase_price_egp": roi_rep.purchase_price_egp,
            "monthly_rent_egp": roi_rep.monthly_rent_egp,
            "gross_yield_pct": roi_rep.gross_rental_yield_pct,
            "net_yield_pct": roi_rep.net_rental_yield_pct,
            "accessibility": geo_prof.accessibility_score,
            "investment_score": roi_rep.investment_score,
            "investment_tier": roi_rep.investment_tier,
        })

    res_df = pd.DataFrame(results)
    sort_col = "gross_yield_pct" if args.sort_by == "yield" else ("investment_score" if args.sort_by == "score" else "purchase_price_egp")
    res_df = res_df.sort_values(sort_col, ascending=False).head(args.top_k)

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() == ".json":
            res_df.to_json(out_path, orient="records", indent=2)
        else:
            res_df.to_csv(out_path, index=False)
        print(f"Exported top {len(res_df)} ranked assets to: {out_path}")

    print("\n" + "=" * 80)
    print(f"             BATCH RANKING RESULTS (SORTED BY: {args.sort_by.upper()})")
    print("=" * 80)
    print(f"{'Rank':<5} {'Location':<25} {'Price (EGP)':<16} {'Yield %':<10} {'Score':<8} {'Tier'}")
    print("-" * 80)
    for rank, (_, row) in enumerate(res_df.iterrows(), 1):
        print(f"#{rank:<4} {row['location'][:23]:<25} {row['purchase_price_egp']:>13,.0f} EGP {row['gross_yield_pct']:>7.2f}% {row['investment_score']:>7.1f} {row['investment_tier']}")
    print("=" * 80)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Property Finder Egypt — Enterprise Geospatial & Valuation CLI")
    parser.add_argument("--model-bundle", default=str(DEFAULT_MODEL_PATH), help="Path to property_price_model_bundle.joblib")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Predict
    p_pred = subparsers.add_parser("predict", help="Estimate asking price with geospatial context")
    p_pred.add_argument("--size", type=float, default=150.0, help="Property built-up area in m²")
    p_pred.add_argument("--bedrooms", type=int, default=3, help="Number of bedrooms")
    p_pred.add_argument("--bathrooms", type=int, default=2, help="Number of bathrooms")
    p_pred.add_argument("--category", choices=["buy", "rent", "commercial_buy", "commercial_rent"], default="buy")
    p_pred.add_argument("--property-type", default="Apartment", help="Property classification")
    p_pred.add_argument("--location", default="Mivida", help="Area or compound name")
    p_pred.add_argument("--market-city", default=None, help="Market city name")
    p_pred.add_argument("--market-region", default=None, help="Market region name")
    p_pred.add_argument("--amenities", type=int, default=5, help="Amenities count")
    p_pred.add_argument("--images", type=int, default=2, help="Listing images count")
    p_pred.add_argument("--lat", type=float, default=None, help="Optional latitude override")
    p_pred.add_argument("--lon", type=float, default=None, help="Optional longitude override")
    p_pred.add_argument("--json", action="store_true", help="Output as JSON")

    # ROI
    p_roi = subparsers.add_parser("roi", help="Calculate real estate yields, cash flows, and investment score")
    p_roi.add_argument("--size", type=float, default=150.0, help="Area in m²")
    p_roi.add_argument("--bedrooms", type=int, default=3, help="Bedrooms")
    p_roi.add_argument("--bathrooms", type=int, default=2, help="Bathrooms")
    p_roi.add_argument("--property-type", default="Apartment")
    p_roi.add_argument("--location", default="Mivida")
    p_roi.add_argument("--market-city", default=None)
    p_roi.add_argument("--market-region", default=None)
    p_roi.add_argument("--amenities", type=int, default=5)
    p_roi.add_argument("--images", type=int, default=2)
    p_roi.add_argument("--lat", type=float, default=None)
    p_roi.add_argument("--lon", type=float, default=None)
    p_roi.add_argument("--purchase-price", type=float, default=None, help="Override purchase price (EGP)")
    p_roi.add_argument("--monthly-rent", type=float, default=None, help="Override monthly rent (EGP)")
    p_roi.add_argument("--down-payment", type=float, default=0.15, help="Down payment fraction (e.g. 0.15 for 15%)")
    p_roi.add_argument("--years", type=int, default=7, help="Installment / loan tenure in years")
    p_roi.add_argument("--interest-rate", type=float, default=0.0, help="Annual interest rate (0.0 for developer off-plan)")
    p_roi.add_argument("--financing", choices=["developer_installment", "mortgage", "cash"], default="developer_installment")
    p_roi.add_argument("--json", action="store_true", help="Output as JSON")

    # Spatial Audit
    p_spatial = subparsers.add_parser("spatial-audit", help="Audit landmark proximity and accessibility score")
    p_spatial.add_argument("--location", default="New Cairo", help="Location text")
    p_spatial.add_argument("--market-city", default=None)
    p_spatial.add_argument("--market-region", default=None)
    p_spatial.add_argument("--lat", type=float, default=None)
    p_spatial.add_argument("--lon", type=float, default=None)
    p_spatial.add_argument("--json", action="store_true", help="Output as JSON")

    # Batch
    p_batch = subparsers.add_parser("batch", help="Batch process and rank multiple listings")
    p_batch.add_argument("--input", required=True, help="Path to input CSV or JSON")
    p_batch.add_argument("--output", default=None, help="Path to export ranked results")
    p_batch.add_argument("--top-k", type=int, default=5, help="Number of listings to display")
    p_batch.add_argument("--sort-by", choices=["yield", "score", "price"], default="yield")

    return parser


def main(args: Optional[List[str]] = None) -> None:
    parser = build_parser()
    parsed_args = parser.parse_args(args)

    dispatch = {
        "predict": cmd_predict,
        "roi": cmd_roi,
        "spatial-audit": cmd_spatial_audit,
        "batch": cmd_batch,
    }
    dispatch[parsed_args.command](parsed_args)


if __name__ == "__main__":
    main()
