from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import pytest

PROJECT_DIR = Path(__file__).resolve().parent.parent


@pytest.fixture(scope="module")
def bundle():
    bundle_path = PROJECT_DIR / "models" / "property_price_model_bundle.joblib"
    return joblib.load(bundle_path)


def test_bundle_keys_and_components(bundle):
    assert "model" in bundle
    assert "location_reference" in bundle
    assert "feature_columns" in bundle
    assert "property_types_by_category" in bundle


def test_location_reference_data(bundle):
    locations = bundle["location_reference"]
    assert isinstance(locations, pd.DataFrame)
    assert len(locations) > 0
    assert "market_region" in locations.columns
    assert "market_city" in locations.columns
    assert "location" in locations.columns


def test_prediction_pipeline_execution(bundle):
    model = bundle["model"]
    feature_columns = bundle["feature_columns"]
    
    # Create valid synthetic feature row
    row = {col: 0 for col in feature_columns}
    if "area_sqm" in row:
        row["area_sqm"] = 150.0
    if "bedrooms" in row:
        row["bedrooms"] = 3
    if "bathrooms" in row:
        row["bathrooms"] = 2
    if "category_buy" in row:
        row["category_buy"] = 1
        
    input_df = pd.DataFrame([row], columns=feature_columns)
    pred = model.predict(input_df)
    
    assert len(pred) == 1
    # Check that predicted price or log-price is a positive real number
    assert pred[0] > 0
