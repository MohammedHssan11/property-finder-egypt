"""
Unit tests for Geospatial Feature Extraction Engine (spatial.py).
"""

import math
import pandas as pd
import pytest

from spatial import (
    LANDMARKS,
    REGIONAL_CENTROIDS,
    GeospatialFeatureExtractor,
    calculate_bearing,
    compute_accessibility_score,
    detect_compound_status,
    haversine_distance,
    manhattan_distance,
)


def test_haversine_distance_properties():
    cairo_cbd = LANDMARKS["cairo_cbd"]
    airport = LANDMARKS["cairo_airport"]

    # 1. Zero distance to itself
    assert haversine_distance(cairo_cbd[0], cairo_cbd[1], cairo_cbd[0], cairo_cbd[1]) == 0.0

    # 2. Symmetry
    d1 = haversine_distance(cairo_cbd[0], cairo_cbd[1], airport[0], airport[1])
    d2 = haversine_distance(airport[0], airport[1], cairo_cbd[0], cairo_cbd[1])
    assert math.isclose(d1, d2, rel_tol=1e-5)

    # 3. Known distance range: Downtown Cairo to Cairo Airport is ~17-21 km
    assert 15.0 <= d1 <= 25.0


def test_calculate_bearing():
    # Point A to Point B directly north
    lat1, lon1 = 30.0, 31.0
    lat2, lon2 = 31.0, 31.0
    bearing_north = calculate_bearing(lat1, lon1, lat2, lon2)
    assert math.isclose(bearing_north, 0.0, abs_tol=1.0) or math.isclose(bearing_north, 360.0, abs_tol=1.0)

    # Point A to Point B directly east
    lat3, lon3 = 30.0, 32.0
    bearing_east = calculate_bearing(lat1, lon1, lat3, lon3)
    assert 88.0 <= bearing_east <= 92.0


def test_manhattan_distance():
    p1 = LANDMARKS["cairo_cbd"]
    p2 = LANDMARKS["new_cairo_golden_square"]
    geo_dist = haversine_distance(p1[0], p1[1], p2[0], p2[1])
    man_dist = manhattan_distance(p1[0], p1[1], p2[0], p2[1])

    # Manhattan is always >= Euclidean/Geodesic distance
    assert man_dist >= geo_dist
    assert man_dist > 0.0


def test_detect_compound_status():
    # Tier 1 master developer compounds
    res1 = detect_compound_status("Penthouse in Mivida, New Cairo")
    assert res1["is_compound"] is True
    assert res1["tier"] == "Tier-1 Master Developer"
    assert "mivida" in res1["matched_pattern"]

    res2 = detect_compound_status("Villa in Palm Hills October")
    assert res2["is_compound"] is True
    assert res2["tier"] == "Tier-1 Master Developer"

    # Generic gated compound
    res3 = detect_compound_status("Duplex in Lake Residence Compound")
    assert res3["is_compound"] is True
    assert res3["tier"] == "Gated Community"

    # Standard urban standalone flat
    res4 = detect_compound_status("Apartment on Talaat Harb Street, Downtown")
    assert res4["is_compound"] is False
    assert res4["tier"] == "Standard Urban / Standalone"
    assert res4["matched_pattern"] is None


def test_accessibility_score_extremes():
    # Prime central location (Downtown Tahrir)
    prime_distances = {
        "cairo_cbd": 0.5,
        "new_capital_financial": 45.0,
        "new_cairo_golden_square": 25.0,
        "sheikh_zayed_central": 28.0,
        "cairo_airport": 18.0,
    }
    score_prime = compute_accessibility_score(prime_distances)
    assert score_prime >= 80.0

    # Very remote desert location
    remote_distances = {
        "cairo_cbd": 250.0,
        "new_capital_financial": 220.0,
        "new_cairo_golden_square": 230.0,
        "sheikh_zayed_central": 260.0,
        "cairo_airport": 240.0,
    }
    score_remote = compute_accessibility_score(remote_distances)
    assert score_remote == 10.0  # Clamped to minimum floor


def test_extractor_imputation_and_profile():
    extractor = GeospatialFeatureExtractor()

    # 1. Test missing coordinates with regional hint
    lat, lon, imputed = extractor.impute_coordinates(None, None, market_region="Alexandria")
    assert imputed is True
    assert math.isclose(lat, REGIONAL_CENTROIDS["alexandria"][0], abs_tol=0.1)
    assert math.isclose(lon, REGIONAL_CENTROIDS["alexandria"][1], abs_tol=0.1)

    # 2. Test valid coordinates inside Egypt bounds
    lat_v, lon_v, imputed_v = extractor.impute_coordinates(30.02, 31.45, market_region="Cairo")
    assert imputed_v is False
    assert lat_v == 30.02
    assert lon_v == 31.45

    # 3. Test extract_features
    profile = extractor.extract_features(
        latitude=30.0270,
        longitude=31.4980,
        location_text="Apartment in Mivida Golden Square",
        market_region="Cairo",
        market_city="New Cairo - El Tagamoa",
    )
    assert profile.is_compound is True
    assert profile.nearest_hub == "new_cairo_golden_square"
    assert profile.nearest_hub_distance_km < 1.0
    assert 0.0 <= profile.accessibility_score <= 100.0

    # 4. Test flat dict export
    d = extractor.to_dict(profile)
    assert "distance_cairo_cbd_km" in d
    assert "distance_new_capital_financial_km" in d
    assert "accessibility_score" in d
    assert "is_compound" in d
    assert d["is_compound"] == 1
