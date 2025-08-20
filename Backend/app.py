from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import uvicorn

# Import the prediction helper from model.py
from model import predict_and_advise

app = FastAPI(title="Screen Time Risk Predictor API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.post("/predict")
async def predict(request: Request):
    """Accepts JSON with screen-time features, returns predicted health impacts and advice.

    Supported keys (snake_case or CSV-style):
    - avg_daily_screen_time_hr or Avg_Daily_Screen_Time_hr (float)
    - exceeded_recommended_limit or Exceeded_Recommended_Limit (int 0/1 or boolean)
    - educational_to_recreational_ratio or Educational_to_Recreational_Ratio (float)
    - age or Age (int)
    """
    payload = await request.json()

    # Mapping for required fields and their alternative CSV-style keys
    field_keys = {
        "avg_daily_screen_time_hr": ["avg_daily_screen_time_hr", "Avg_Daily_Screen_Time_hr"],
        "exceeded_recommended_limit": ["exceeded_recommended_limit", "Exceeded_Recommended_Limit"],
        "educational_to_recreational_ratio": ["educational_to_recreational_ratio", "Educational_to_Recreational_Ratio"],
        "age": ["age", "Age"],
    }

    features = {}
    missing = []

    for canonical, options in field_keys.items():
        found = False
        for key in options:
            if key in payload:
                features[canonical] = payload[key]
                found = True
                break
        if not found:
            missing.append(options[0])

    if missing:
        raise HTTPException(status_code=422, detail={
            "error": "missing_fields",
            "missing": missing,
            "message": "Required fields are missing. Accepts CSV-style or snake_case keys."
        })

    # Coerce and validate types
    try:
        features["avg_daily_screen_time_hr"] = float(features["avg_daily_screen_time_hr"])
        # Exceeded limit may be boolean or 0/1; coerce to int 0/1
        val = features["exceeded_recommended_limit"]
        if isinstance(val, bool):
            features["exceeded_recommended_limit"] = int(val)
        else:
            features["exceeded_recommended_limit"] = int(val)
        if features["exceeded_recommended_limit"] not in (0, 1):
            raise ValueError("exceeded_recommended_limit must be 0 or 1")

        features["educational_to_recreational_ratio"] = float(features["educational_to_recreational_ratio"])
        features["age"] = int(features["age"])
    except Exception as e:
        raise HTTPException(status_code=422, detail={
            "error": "invalid_field_types",
            "message": str(e)
        })

    # Call the model helper which loads model artifacts and returns display labels + advice
    try:
        predicted_labels, advice = predict_and_advise(features)
    except Exception as e:
        raise HTTPException(status_code=500, detail={
            "error": "prediction_failed",
            "message": str(e)
        })

    return {
        "predicted_health_impacts": predicted_labels,
        "personalized_advice": advice,
    }


if __name__ == "__main__":
    # Run with uvicorn when executed directly
    uvicorn.run("app:app", host="0.0.0.0", port=8000)
