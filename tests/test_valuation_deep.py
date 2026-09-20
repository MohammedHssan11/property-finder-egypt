"""
Deep integration tests for Property Finder valuation, spatial enrichment, and batch analytics.
"""

import json
from pathlib import Path
import pandas as pd
import pytest

import cli
from evals import runner
from roi import RealEstateROIEngine
from spatial import GeospatialFeatureExtractor


@pytest.fixture(scope="module")
def bundle():
    return cli.load_bundle()


def test_dual_valuation_with_real_bundle(bundle):
    roi_engine = RealEstateROIEngine(bundle)

    # 1. Residential Apartment in New Cairo
    res_input = {
        "size": 180.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "property_type": "Apartment",
        "category": "buy",
        "market_region": "Cairo",
        "market_city": "New Cairo - El Tagamoa",
        "local_area": "Mivida",
        "location": "Mivida, New Cairo",
        "amenities_count": 6,
        "images_count": 4,
        "latitude": 30.0270,
        "longitude": 31.4980,
    }
    buy_price, rent_price = roi_engine.estimate_dual_valuation(res_input)
    assert buy_price > 1_000_000.0
    assert rent_price > 10_000.0

    # 2. Commercial Office Space in New Capital
    comm_input = {
        "size": 120.0,
        "bedrooms": 0,
        "bathrooms": 1,
        "property_type": "Office Space",
        "category": "commercial_buy",
        "market_region": "Cairo",
        "market_city": "New Capital City",
        "local_area": "Financial District",
        "location": "Financial District, New Capital",
        "amenities_count": 8,
        "images_count": 5,
        "latitude": 30.0167,
        "longitude": 31.7500,
    }
    c_buy, c_rent = roi_engine.estimate_dual_valuation(comm_input)
    assert c_buy > 500_000.0
    assert c_rent > 5_000.0


def test_spatial_extractor_reference_density(bundle):
    locations_df = bundle["location_reference"]
    extractor = GeospatialFeatureExtractor(location_reference=locations_df)

    # Search for an area with multiple listings
    profile = extractor.extract_features(
        latitude=30.0270,
        longitude=31.4980,
        location_text="New Cairo",
        market_region="Cairo",
        market_city="New Cairo - El Tagamoa",
    )
    assert profile.local_density_count > 0
    assert profile.local_median_sqm_egp is not None
    assert profile.local_median_sqm_egp > 1_000.0


def test_batch_json_pipeline(tmp_path, capsys):
    json_in = tmp_path / "listings.json"
    json_out = tmp_path / "ranked_out.json"

    records = [
        {"location": "Mivida, New Cairo", "size": 160.0, "bedrooms": 3, "bathrooms": 2, "property_type": "Apartment"},
        {"location": "Arkan, Sheikh Zayed", "size": 240.0, "bedrooms": 4, "bathrooms": 3, "property_type": "Villa"},
    ]
    json_in.write_text(json.dumps(records), encoding="utf-8")

    cli.main(["batch", "--input", str(json_in), "--output", str(json_out), "--sort-by", "score", "--top-k", "2"])
    captured = capsys.readouterr()
    assert "BATCH RANKING RESULTS" in captured.out
    assert json_out.exists()

    data = json.loads(json_out.read_text(encoding="utf-8"))
    assert len(data) == 2
    assert "investment_score" in data[0]


def test_evals_runner_cli(tmp_path, capsys):
    out_rep = tmp_path / "CLI_BENCHMARK.md"
    runner.main(args=["--output", str(out_rep)])
    captured = capsys.readouterr()
    assert "BENCHMARK EVALUATION SUMMARY" in captured.out
    assert "Properties Evaluated" in captured.out
    assert out_rep.exists()


def test_cli_error_handling():
    with pytest.raises(FileNotFoundError):
        cli.main(["batch", "--input", "nonexistent_file_xyz_123.csv"])
