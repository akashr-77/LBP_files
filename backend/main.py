# main.py
# FastAPI application: defines routes, wires together model + features + schemas.
# Start with:  uvicorn main:app --reload --port 8000

import logging
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas  import (signalInput, FaultPredictionResponse,
                       ModelStatusResponse)
from model    import bearing_model
from features import (extract_windows, normalize_windows,
                      reshape, MODEL_CLASSES)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
)
logger = logging.getLogger(__name__)

START_TIME = time.time()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup: runs before the first request ──
    logger.info("=== Backend starting up ===")
    try:
        bearing_model.load()          # loads best_cwru_cnn.keras into memory
        logger.info("=== Model ready. Accepting requests. ===")
    except FileNotFoundError as e:
        logger.error(f"STARTUP FAILED: {e}")
        # App still starts but /health will report model_loaded=False
    yield
    # ── Shutdown: runs when process ends ──
    logger.info("=== Backend shutting down ===")

    # ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Bearing Fault Detection API",
    description="CWRU vibration signal analysis — fault classification & RUL estimation",
    version="1.0.0",
    lifespan=lifespan,
)

# Allow any frontend origin during local testing
# In production: replace "*" with your frontend's URL
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

#Route 1: Model status check
@app.get("/model_status", response_model=ModelStatusResponse, tags=["Model_Status"])
def model_status():
    """
    Check if the model is loaded and ready to accept requests.
    Useful for frontend health checks and monitoring.
    """
    if bearing_model.is_loaded:
        uptime_seconds = round(time.time() - START_TIME, 1)
        return ModelStatusResponse(
            status="ok",
            model_loaded=True,
            uptime_seconds=uptime_seconds,
        )
    else:
        return ModelStatusResponse(
            status="degraded",
            model_loaded=False,
            # model_name=None,
        )
    
#Route 2: Fault class prediction
@app.post("/predict_fault", response_model=FaultPredictionResponse, tags=["Fault_Prediction"])
def predict_fault(input_data: signalInput):
    """
    Predict the fault class from the input signal.
    Steps:
    1. Extract overlapping windows from the raw signal.
    2. Normalize each window to match training conditions.
    3. Run inference on each window and average results.
    4. Return the predicted class, confidence, and probabilities.
    """
    if not bearing_model.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded. Try again later.")

    try:
        windows = extract_windows(input_data.signal)
        normalized_windows = normalize_windows(windows)
        reshaped_windows = reshape(normalized_windows) # add channel dimension if needed
        probabilities = bearing_model.predict_proba(reshaped_windows)
        avg_probabilities = probabilities.mean(axis=0)
        predicted_index = int(avg_probabilities.argmax())
        predicted_class = MODEL_CLASSES[predicted_index]
        # confidence = float(avg_probabilities[predicted_index])
        class_prob_dict = {MODEL_CLASSES[i]: float(avg_probabilities[i]) for i in range(len(MODEL_CLASSES))}

        return FaultPredictionResponse(
            fault_class=predicted_class,
            fault_code=predicted_index,
            # confidence=confidence,
            class_probabilities=class_prob_dict,
            window_used=windows.shape[0],
            preprocessing_note="Signal was windowed and normalized before prediction."
        )
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        raise HTTPException(status_code=500, detail="An error occurred during prediction. See logs for details.")
    
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)