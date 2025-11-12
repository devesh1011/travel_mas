from pydantic import BaseModel, Field
from typing import Optional
from google.genai import types


class POI(BaseModel):
    """A Point Of Interest suggested by the agent."""

    place_name: str = Field(description="Name of the attraction")
    address: str = Field(
        description="An address or sufficient information to geocode for a Lat/Lon"
    )
    lat: str = Field(
        description="Numerical representation of Latitude of the location (e.g., 20.6843)"
    )
    long: str = Field(
        description="Numerical representation of Longitude of the location (e.g., -88.5678)"
    )
    review_ratings: str = Field(
        description="Numerical representation of rating (e.g. 4.8 , 3.0 , 1.0 etc)"
    )
    highlights: str = Field(description="Short description highlighting key features")
    image_url: str = Field(description="verified URL to an image of the destination")
    map_url: Optional[str] = Field(description="Verified URL to Google Map")
    place_id: Optional[str] = Field(description="Google Map place_id")


class POISuggestions(BaseModel):
    """Points of interest recommendation."""

    places: list[POI]


class Destination(BaseModel):
    """A destination recommendation."""

    name: str = Field(description="A Destination's Name")
    country: str = Field(description="The Destination's Country Name")
    image: str = Field(description="verified URL to an image of the destination")
    highlights: str = Field(description="Short description highlighting key features")
    rating: str = Field(description="Numerical rating (e.g., 4.5)")


class DestinationIdeas(BaseModel):
    """Destinations recommendation."""

    places: list[Destination]


# Convenient declaration for controlled generation.
json_response_config = types.GenerateContentConfig(
    response_mime_type="application/json"
)
