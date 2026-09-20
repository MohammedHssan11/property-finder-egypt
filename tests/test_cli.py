"""
Unit tests for Property Finder Enterprise CLI (cli.py).
"""

import json
from pathlib import Path
import pandas as pd
import pytest

import cli


def test_cli_predict_stdout(capsys):
    cli.main(["predict", "--size", "160", "--bedrooms", "3", "--bathrooms", "2", "--location", "Mivida"])
    captured = capsys.readouterr()
    assert "PROPERTY FINDER EGYPT — PRICE ESTIMATION" in captured.out
    assert "ESTIMATED VALUATION" in captured.out
    assert "SPATIAL & PROXIMITY TELEMETRY" in captured.out


def test_cli_predict_json(capsys):
    cli.main(["predict", "--size", "140", "--bedrooms", "2", "--bathrooms", "2", "--location", "Rehab", "--json"])
    captured = capsys.readouterr()
    data = json.loads(captured.out)
    assert "predicted_price_egp" in data
    assert data["predicted_price_egp"] > 0
    assert "geospatial" in data
    assert "accessibility_score" in data["geospatial"]


def test_cli_roi_stdout_and_json(capsys):
    # Stdout mode
    cli.main([
        "roi",
        "--size", "150",
        "--bedrooms", "3",
        "--location", "New Cairo",
        "--financing", "developer_installment",
        "--years", "7"
    ])
    captured = capsys.readouterr()
    assert "REAL ESTATE ROI AUDIT" in captured.out
    assert "YIELD & PERFORMANCE METRICS" in captured.out
    assert "5-YEAR CAPITAL APPRECIATION" in captured.out

    # JSON mode
    cli.main(["roi", "--size", "150", "--bedrooms", "3", "--location", "New Cairo", "--json"])
    captured_json = capsys.readouterr()
    roi_data = json.loads(captured_json.out)
    assert "gross_rental_yield_pct" in roi_data
    assert "net_rental_yield_pct" in roi_data
    assert "investment_score" in roi_data


def test_cli_spatial_audit(capsys):
    cli.main(["spatial-audit", "--location", "Sheikh Zayed"])
    captured = capsys.readouterr()
    assert "GEOSPATIAL SPATIAL AUDIT" in captured.out
    assert "LANDMARK DISTANCES" in captured.out

    cli.main(["spatial-audit", "--location", "Sheikh Zayed", "--json"])
    captured_json = capsys.readouterr()
    audit_data = json.loads(captured_json.out)
    assert "nearest_hub" in audit_data
    assert "accessibility_score" in audit_data


def test_cli_batch(tmp_path, capsys):
    input_file = tmp_path / "test_listings.csv"
    output_file = tmp_path / "ranked_listings.csv"

    sample_df = pd.DataFrame([
        {"location": "New Cairo", "size": 150.0, "bedrooms": 3, "bathrooms": 2, "property_type": "Apartment"},
        {"location": "Sheikh Zayed", "size": 220.0, "bedrooms": 4, "bathrooms": 3, "property_type": "Villa"},
        {"location": "Downtown Cairo", "size": 110.0, "bedrooms": 2, "bathrooms": 1, "property_type": "Apartment"},
    ])
    sample_df.to_csv(input_file, index=False)

    cli.main(["batch", "--input", str(input_file), "--output", str(output_file), "--sort-by", "yield", "--top-k", "3"])
    captured = capsys.readouterr()
    assert "BATCH RANKING RESULTS" in captured.out
    assert "#1" in captured.out
    assert output_file.exists()

    ranked_df = pd.read_csv(output_file)
    assert len(ranked_df) == 3
    assert "gross_yield_pct" in ranked_df.columns
    assert "investment_score" in ranked_df.columns
