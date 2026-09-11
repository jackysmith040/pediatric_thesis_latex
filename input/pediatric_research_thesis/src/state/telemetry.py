from pydantic import BaseModel, Field

class TelemetryState(BaseModel):
    current_children: int = Field(default=0, description="Current number of children in view")
    current_adults: int = Field(default=0, description="Current number of adults in view")
    total_daily_children: int = Field(default=0, description="Total children processed today")
    total_daily_adults: int = Field(default=0, description="Total adults processed today")
    overcrowding_alert: bool = Field(default=False, description="Whether overcrowding alert is active")
