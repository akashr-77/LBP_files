# schemas.py
# Pydantic models define the "contract" between frontend and backend.
# FastAPI reads these and auto-validates every incoming request.
# If frontend sends wrong types or missing fields → 422 error before model is touched.

from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
from datetime import datetime
import math

#Inputs 
#Field is used to add extra validation and metadata to the model fields.
class signalInput(BaseModel):
    signal: List[float] = Field(
        ..., # Ellipsis indicates that this field is required.
        description="The input signal as a list of floats.",
        min_length = 1024, # Minimum length of the signal list.
    )
    sampling_rate: Optional[int] = Field(
        default = 48000, #CWRU default sampling rate is 48kHz.
        description="The sampling rate of the input signal in Hz. Default is 48000 Hz.",
        ge = 1000, # Minimum sampling rate of 1000 Hz.
        le = 100000, # Maximum sampling rate of 100000 Hz.
    )

    sensor_id: Optional[str] = Field(
        default = "DE",
        description="The sensor ID from which the signal was recorded (Drive End or Fan End). Default is 'DE'.",
    ) 

    #check for NaN or Inf values in the signal list after type check.
    @field_validator('signal') #default mode is "after"
    @classmethod
    def no_nan_or_inf(cls, v):
        if any(math.isnan(x) or math.isinf(x) for x in v):
            raise ValueError("Signal contains NaN or infinite values.")
        return v


#Responses 
#Fault class prediction
class FaultPredictionResponse(BaseModel):
    fault_class: str  #e.g. "Inner race fault"
    fault_code: int   #0=Normal 
    confidence: Optional[float] = None #0.0 to 1.0
    class_probabilities: Optional[Dict[str, float]] = None #e.g. {"Normal": 0.1, "Fault": 0.9}
    window_used: Optional[int] = None #windw size 
    preprocessing_note: Optional[str] = None #e.g. "Signal was normalized before prediction."


#RUL estimation 
# class RULPredictionResponse(BaseModel):
#     rul_estimate_cycle: float #Estimated RUL in cycles
#     rul_estimate_hours: float #Estimated RUL in hours
#     severity: str #e.g. "Low", "Medium", "High"
#     degradation_note: Optional[str] = None #e.g. "RUL estimate is based on current operating conditions and may change with new data."

#Model status
class ModelStatusResponse(BaseModel):
    status: str  #ok or degraded
    model_loaded: bool
    model_name: Optional[str] = None
    model_path: Optional[str] = None
    timestamp: Optional[str] = None
    uptime_seconds: Optional[float] = None #Time since model was loaded


#Chat (Gemini)
class ChatRequest(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        description="The user's chat message.",
    )

class ChatResponse(BaseModel):
    reply: str  # Gemini's response text
