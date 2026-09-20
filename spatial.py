"""
Geospatial Feature Extraction Engine for Egyptian Real Estate.

Computes geodesic distances to primary economic/business hubs, directional bearings,
accessibility indices, gated compound classifications, and local spatial density benchmarks.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


EARTH_RADIUS_KM = 6371.0


LANDMARKS: Dict[str, Tuple[float, float]] = {
    "cairo_cbd": (30.0444, 31.2357),          # Downtown Tahrir & Nile Corniche
    "new_capital_financial": (30.0167, 31.7500), # New Administrative Capital / Iconic Tower
    "new_cairo_golden_square": (30.0270, 31.4980), # 5th Settlement Golden Square & AUC
    "sheikh_zayed_central": (30.0350, 30.9850), # Sheikh Zayed Arkan & 26th July Corridor
    "cairo_airport": (30.1219, 31.4056),      # Cairo International Airport
    "alexandria_corniche": (31.2001, 29.9187), # Mediterranean Hub / Alexandria Corniche
    "red_sea_hub": (27.2579, 33.8116),         # Hurghada & El Gouna Marina
    "north_coast_hub": (30.9200, 28.7500),     # New Alamein & Sahel Corridor
}

# Regional fallback centroids when latitude/longitude are missing
REGIONAL_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "cairo": (30.0444, 31.2357),
    "giza": (30.0131, 31.2089),
    "alexandria": (31.2001, 29.9187),
    "red sea": (27.2579, 33.8116),
    "south sinai": (27.9158, 34.3299),
    "al behere": (31.0379, 30.4682),
    "matrouh": (31.3543, 27.2373),
    "dakahlia": (31.0409, 31.3785),
    "sharqia": (30.5877, 31.5020),
    "gharbia": (30.7865, 31.0004),
    "qalyubia": (30.3292, 31.2168),
    "suez": (29.9668, 32.5498),
    "ismailia": (30.5965, 32.2715),
    "port said": (31.2653, 32.3019),
}

# Master Developer & Gated Community Keyword Patterns
PREMIUM_DEVELOPERS = [
    r"\bemaar\b", r"\bmivida\b", r"\bmarassi\b", r"\buptown cairo\b",
    r"\bpalm hills\b", r"\bbadya\b", r"\bhacienda\b",
    r"\bsodic\b", r"\ballegria\b", r"\bvye\b", r"\beastown\b", r"\bvillette\b",
    r"\btalaat moustafa\b", r"\bmadinaty\b", r"\bal rehab\b", r"\bcelia\b",
    r"\bmountain view\b", r"\bi-city\b", r"\bchillout park\b",
    r"\bhyde park\b", r"\bkatameya\b", r"\bkatameya heights\b", r"\bkatameya dunes\b",
    r"\bswan lake\b", r"\bhassan allam\b", r"\bhaptown\b",
    r"\bal burouj\b", r"\btaj city\b", r"\bsarraf\b",
    r"\bel gouna\b", r"\bsomabay\b", r"\bsahl hasheesh\b",
    r"\bil monte galala\b", r"\bzed\b", r"\bora\b"
]


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Computes exact geodesic distance between two coordinate pairs in kilometers.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (
        math.sin(delta_phi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2
    )
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))
    return round(EARTH_RADIUS_KM * c, 3)


def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculates initial compass bearing from point 1 to point 2 in degrees [0, 360).
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_lambda = math.radians(lon2 - lon1)

    y = math.sin(delta_lambda) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(delta_lambda)

    theta = math.atan2(y, x)
    bearing_deg = (math.degrees(theta) + 360.0) % 360.0
    return round(bearing_deg, 2)


def manhattan_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Approximates grid (Manhattan) transit distance in kilometers.
    """
    lat_km = abs(lat2 - lat1) * 111.0
    mean_lat_rad = math.radians((lat1 + lat2) / 2.0)
    lon_km = abs(lon2 - lon1) * 111.0 * math.cos(mean_lat_rad)
    return round(lat_km + lon_km, 3)


def detect_compound_status(text: str) -> Dict[str, Any]:
    """
    Detects whether a location or description belongs to a gated community or master-planned compound.
    """
    cleaned = str(text or "").lower()
    for pattern in PREMIUM_DEVELOPERS:
        if re.search(pattern, cleaned):
            return {
                "is_compound": True,
                "tier": "Tier-1 Master Developer",
                "matched_pattern": pattern.strip(r"\b"),
            }

    generic_terms = ["compound", "residence", "residences", "gated", "park", "city", "heights", "palace"]
    for term in generic_terms:
        if term in cleaned:
            return {
                "is_compound": True,
                "tier": "Gated Community",
                "matched_pattern": term,
            }

    return {
        "is_compound": False,
        "tier": "Standard Urban / Standalone",
        "matched_pattern": None,
    }


def compute_accessibility_score(distances: Dict[str, float]) -> float:
    """
    Calculates an accessibility index [10.0, 100.0] based on proximity to major CBDs and airport.
    """
    d_cairo = distances.get("cairo_cbd", 50.0)
    d_nac = distances.get("new_capital_financial", 60.0)
    d_golden = distances.get("new_cairo_golden_square", 40.0)
    d_zayed = distances.get("sheikh_zayed_central", 50.0)
    d_airport = distances.get("cairo_airport", 45.0)

    # Nearest major business district
    d_nearest_cbd = min(d_cairo, d_nac, d_golden, d_zayed)

    penalty = (0.75 * d_nearest_cbd) + (0.35 * d_airport)
    score = 100.0 - penalty
    return round(float(np.clip(score, 10.0, 100.0)), 1)


@dataclass
class GeospatialProfile:
    latitude: float
    longitude: float
    coordinate_imputed: bool
    landmark_distances_km: Dict[str, float]
    nearest_hub: str
    nearest_hub_distance_km: float
    accessibility_score: float
    is_compound: bool
    compound_tier: str
    matched_compound: Optional[str]
    local_density_count: int
    local_median_sqm_egp: Optional[float]


class GeospatialFeatureExtractor:
    """
    Geospatial Feature Extractor for Egyptian Real Estate.
    """

    def __init__(self, location_reference: Optional[pd.DataFrame] = None):
        self.location_reference = location_reference

    def impute_coordinates(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        market_region: Optional[str] = None,
        market_city: Optional[str] = None,
    ) -> Tuple[float, float, bool]:
        """
        Returns validated coordinates or centroid imputation when missing.
        """
        if latitude is not None and longitude is not None:
            if not pd.isna(latitude) and not pd.isna(longitude):
                # Bounds check for Egypt approx [22.0, 32.0] N and [24.0, 37.0] E
                if 21.0 <= latitude <= 33.0 and 23.0 <= longitude <= 38.0:
                    return float(latitude), float(longitude), False

        # Try regional centroid
        region_key = str(market_region or "").lower().strip()
        for k, coords in REGIONAL_CENTROIDS.items():
            if k in region_key:
                return coords[0], coords[1], True

        city_key = str(market_city or "").lower().strip()
        for k, coords in REGIONAL_CENTROIDS.items():
            if k in city_key:
                return coords[0], coords[1], True

        # Default: Cairo center
        return REGIONAL_CENTROIDS["cairo"][0], REGIONAL_CENTROIDS["cairo"][1], True

    def extract_features(
        self,
        latitude: Optional[float],
        longitude: Optional[float],
        location_text: str = "",
        market_region: Optional[str] = None,
        market_city: Optional[str] = None,
    ) -> GeospatialProfile:
        """
        Extracts complete geospatial profile and landmark distances.
        """
        lat, lon, imputed = self.impute_coordinates(latitude, longitude, market_region, market_city)

        # Distances to all defined landmarks
        distances = {}
        for name, (l_lat, l_lon) in LANDMARKS.items():
            distances[name] = haversine_distance(lat, lon, l_lat, l_lon)

        # Nearest hub
        nearest_hub = min(distances.keys(), key=lambda k: distances[k])
        nearest_hub_dist = distances[nearest_hub]

        # Accessibility score
        accessibility = compute_accessibility_score(distances)

        # Compound detection
        compound_info = detect_compound_status(f"{location_text} {market_city or ''}")

        # Local density and median price benchmark from reference table if present
        density_count = 0
        local_median_sqm = None
        if self.location_reference is not None and not self.location_reference.empty:
            loc_lower = str(location_text).lower().strip()
            matches = self.location_reference[
                self.location_reference["location"].str.lower().str.contains(loc_lower, regex=False, na=False)
            ]
            if not matches.empty:
                density_count = int(matches["listing_count"].sum())
                if "median_price" in matches.columns:
                    local_median_sqm = float(matches["median_price"].median() / 150.0)

        return GeospatialProfile(
            latitude=lat,
            longitude=lon,
            coordinate_imputed=imputed,
            landmark_distances_km=distances,
            nearest_hub=nearest_hub,
            nearest_hub_distance_km=nearest_hub_dist,
            accessibility_score=accessibility,
            is_compound=compound_info["is_compound"],
            compound_tier=compound_info["tier"],
            matched_compound=compound_info["matched_pattern"],
            local_density_count=density_count,
            local_median_sqm_egp=local_median_sqm,
        )

    def to_dict(self, profile: GeospatialProfile) -> Dict[str, Any]:
        """Converts profile to a flat dictionary suitable for DataFrame or model input."""
        res = {
            "latitude": profile.latitude,
            "longitude": profile.longitude,
            "coordinate_imputed": int(profile.coordinate_imputed),
            "accessibility_score": profile.accessibility_score,
            "is_compound": int(profile.is_compound),
            "compound_tier": profile.compound_tier,
            "nearest_hub": profile.nearest_hub,
            "nearest_hub_distance_km": profile.nearest_hub_distance_km,
        }
        for name, d in profile.landmark_distances_km.items():
            res[f"distance_{name}_km"] = d
        return res
