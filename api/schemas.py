from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, Field, model_validator

# Enums - constrain inputs to valid values


class AirlineEnum(StrEnum):
    indigo = "IndiGo"
    air_india = "Air India"
    jet_airways = "Jet Airways"
    spicejet = "SpiceJet"
    goair = "GoAir"
    vistara = "Vistara"
    air_asia = "Air Asia"
    other = "Other"


class CityEnum(StrEnum):
    bangalore = "Bangalore"
    delhi = "Delhi"
    mumbai = "Mumbai"
    chennai = "Chennai"
    kolkata = "Kolkata"
    cochin = "Cochin"
    hyderabad = "Hyderabad"


class StopsEnum(StrEnum):
    non_stop = "non-stop"
    one_stop = "1 stop"
    two_stops = "2 stops"
    three_stops = "3 stops"
    four_stops = "4 stops"


# Request Schema:
class FlightPredictionRequest(BaseModel):
    airline: AirlineEnum
    date_of_journey: str = Field(
        ..., description="Journey date (DD/MM/YYYY or YYYY-MM-DD)"
    )
    source: CityEnum
    destination: CityEnum
    dep_time: str = Field(..., description="Departure time (HH:MM)")
    arrival_time: str = Field(..., description="Arrival time (HH:MM)")
    duration: str = Field(..., description="Duration string like '2h 45m' or '45m'")
    total_stops: StopsEnum
    additional_info: str = Field(default="No info")

    @model_validator(mode="after")
    def validate_source_destination(self) -> FlightPredictionRequest:
        if self.source == self.destination:
            raise ValueError(
                f"Source and destination cannot be the same: {self.source.value}"
            )
        return self

    def to_dataframe(self):
        """Convert request to DataFrame for feature engineering pipeline."""
        import pandas as pd

        return pd.DataFrame(
            [
                {
                    "Airline": self.airline.value,
                    "Date_of_Journey": self.date_of_journey,
                    "Source": self.source.value,
                    "Destination": self.destination.value,
                    "Route": f"{self.source.value} → {self.destination.value}",
                    "Dep_Time": self.dep_time,
                    "Arrival_Time": self.arrival_time,
                    "Duration": self.duration,
                    "Total_Stops": self.total_stops.value,
                    "Additional_Info": self.additional_info,
                }
            ]
        )


## Response Schema:
class FlightPredictionResponse(BaseModel):
    predicted_price_inr: float = Field(..., description="predicted price in INR")
    predictor_type: str = Field(..., description="Which predictor was used")
    status: str = Field(default="success")
    request_id: str = Field(..., description="Correlation ID for this request")


class HealthResponse(BaseModel):
    status: Literal["healthy", "degraded", "unhealthy"]
    model_loaded: bool
    predictor_type: str
    version: str


class ModelInferenceResponse(BaseModel):
    r2_inr_space: float
    rmse_inr: float
    mae_inr: float
    feature_count: int
    train_size: int
    r2_threshold: float
