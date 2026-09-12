"""
Unit tests for feature engineering functions.
These test pure functions — no I/O, no model loading.
Fast to run: ~0.1 seconds total.
"""
import pytest

from flight_predictor.features import(
    parse_duration_to_minutes,
    parse_stops,
    engineer_features,
    compute_route_distance,
    get_airport_coords
)

# parse_duration_to_minutes:
class TestParseDuration:
    def test_hour_and_minutes(self):
        assert parse_duration_to_minutes("2h 45m") == 165

    def test_hours_only(self):
        assert parse_duration_to_minutes("2h") == 120

    def test_minutes_only(self):
        assert parse_duration_to_minutes("45") == 45

    def test_already_numeric(self):
        assert parse_duration_to_minutes("150") == 150
    
    def test_large_duration(self):
        assert parse_duration_to_minutes("12h 30m") == 750
    
    def test_short_flight(self):
        assert parse_duration_to_minutes("50m") == 50

    def test_string_coercion(self):
        assert parse_duration_to_minutes(165) == 165 # integer input

## parse_stops:
class TestParseStops:
    def test_non_stops(self):
        assert parse_stops("non-stops") == 0
    
    def test_one_stop(self):
        assert parse_stops("1 stop") == 1
    
    def test_two_stops(self):
        assert parse_stops("2 stops") == 2

    def test_case_insensitive(self):
        assert parse_stops("Non-Stop") == 0

## get_airport_coords
class TestAirportCoords:
    def test_known_airport(self):
        lat, lon = get_airport_coords("BLR")
        assert abs(lat - 12.9716) < 0.01
        assert abs(lon - 77.5946) < 0.01

    def test_unkown_airport_fallback(self):
        lat, lon = get_airport_coords("XYZ")
        assert lat == 0.0
        assert lon == 0.0

    def test_case_insensitive(self):
        lat1, _ = get_airport_coords("DEL")
        lat2, _ = get_airport_coords("del")
        assert lat1 == lat2

## engineer_features:
class TestEngineerFeatures:
    def test_returns_new_dataframe(self, sample_df):
        """engineer_features should not modify the original DataFrame."""
        original_cols = list(sample_df.columns)
        result = engineer_features(sample_df)
        assert list(sample_df.columns) == original_cols
        assert "duration_minutes" in result.columns

    def test_duration_minutes_correct(self, sample_df):
        result = engineer_features(sample_df)
        assert result["duration_minutes"].iloc[0] == 165

    def test_data_features_extracted(self, sample_df):
        result = engineer_features(sample_df)
        assert "journey_day" in result.columns
        assert "journey_month" in result.columns
        assert "dep_hour" in result.columns

    def test_price_column_not_required(self, sample_df):
        df_no_price = sample_df.drop(columns = ["Price"], errors="ignore")
        result = engineer_features(df_no_price)
        assert "duration_minutes" in result.columns

    def test_short_flight(self, sample_batch_df):
        result = engineer_features(sample_batch_df)
        short_flight_mask = sample_batch_df["Duration"] == "45m"
        short_flight_duration = result.loc[short_flight_mask, "duration_minutes"]
        assert (short_flight_duration == 45).all()