import uvicorn
from fastapi import FastAPI, HTTPException
from pathlib import Path
from models.passenger import Passenger
from libs.model import train, predict

BASE_DIR = Path(__file__).resolve(strict=True).parent
MODEL_DIR = Path(BASE_DIR).joinpath("ml_models")
DATA_DIR = Path(BASE_DIR).joinpath("data")

app = FastAPI(title="Titanic Survival API")


@app.get("/", tags=["intro"])
async def index():
    return {"message": "API do przewidywania przeżycia na Titanicu (RandomForest)"}


@app.post("/model/train", tags=["model"])
async def train_model(filename: str = "DSP_6.csv"):
    data_file = Path(DATA_DIR).joinpath(filename)
    model_file = Path(MODEL_DIR).joinpath("titanic_forest.pkl")

    if not data_file.exists():
        raise HTTPException(status_code=404, detail=f"Plik {filename} nie został znaleziony w folderze /data")

    try:
        accuracy = train(data_file, model_file)
        return {
            "message": "Model wytrenowany i zapisany pomyślnie",
            "model_path": str(model_file),
            "training_accuracy": accuracy
        }
    except Exception as e:
        print(f"Błąd treningu: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/model/predict", tags=["model"])
async def predict_survival(passenger: Passenger):
    model_file = Path(MODEL_DIR).joinpath("titanic_forest.pkl")

    if not model_file.exists():
        raise HTTPException(status_code=404,
                            detail="Model nie znaleziony. Najpierw wytrenuj model używając endpointu /train.")

    try:
        data = passenger.model_dump()

        result = predict(data, model_file)

        outcome = "Przeżył" if result == 1 else "Nie przeżył"
        return {
            "prediction": result,
            "outcome": outcome,
            "passenger_data": data
        }
    except Exception as e:
        print(f"Błąd predykcji: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8008)