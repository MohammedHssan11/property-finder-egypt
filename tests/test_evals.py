"""
Unit tests for Property Finder Benchmark Evaluation Suite (evals/runner.py).
"""

from pathlib import Path
import pytest

from evals.runner import BENCHMARK_PROPERTIES, generate_markdown_report, run_benchmark_evaluations


def test_benchmark_properties_dataset():
    assert len(BENCHMARK_PROPERTIES) == 8
    for item in BENCHMARK_PROPERTIES:
        assert "id" in item
        assert "market" in item
        assert "size" in item and item["size"] > 0
        assert "bedrooms" in item
        assert "min_yield_pct" in item and "max_yield_pct" in item
        assert item["min_yield_pct"] < item["max_yield_pct"]


def test_run_benchmark_evaluations_end_to_end(tmp_path):
    report_file = tmp_path / "TEST_BENCHMARK_RESULTS.md"
    results = run_benchmark_evaluations(report_path=report_file)

    assert "num_properties" in results
    assert results["num_properties"] == 8
    assert "hub_accuracy_pct" in results
    assert results["hub_accuracy_pct"] >= 75.0
    assert "mean_gross_yield_pct" in results
    assert 3.0 <= results["mean_gross_yield_pct"] <= 15.0
    assert "cases" in results
    assert len(results["cases"]) == 8

    # Verify report file generation
    assert report_file.exists()
    content = report_file.read_text(encoding="utf-8")
    assert "Property Finder Egypt — Geospatial & ROI Benchmark Report" in content
    assert "M1_NEW_CAIRO" in content
    assert "M3_SHEIKH_ZAYED" in content


def test_generate_markdown_report(tmp_path):
    out_file = tmp_path / "dummy_report.md"
    dummy_results = {
        "num_properties": 1,
        "hub_accuracy_pct": 100.0,
        "yield_sanity_pass_rate_pct": 100.0,
        "mean_gross_yield_pct": 8.5,
        "mean_accessibility_score": 75.0,
        "mean_investment_score": 82.0,
        "cases": [
            {
                "id": "T1",
                "market": "Test Prime Tagamoa",
                "purchase_price_egp": 10000000.0,
                "monthly_rent_egp": 70000.0,
                "gross_yield_pct": 8.4,
                "nearest_hub": "new_cairo_golden_square",
                "hub_dist_km": 3.2,
                "tier": "A (Strong Growth)",
            }
        ],
    }

    generate_markdown_report(dummy_results, out_file)
    assert out_file.exists()
    text = out_file.read_text(encoding="utf-8")
    assert "Executive Performance Matrix" in text
    assert "Test Prime Tagamoa" in text
