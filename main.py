# FastAPI - Swagger UI => http://localhost:8000/docs

from fastapi import FastAPI, UploadFile, File, HTTPException
from prediction import predict_text, predict_file

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Bienvenue dans l'API d'analyse des sentiments 👀"}

@app.post("/predict/")
async def predict_text_endpoint(text: str):
    try:
        sentiment = predict_text(text)
        return {"sentiment": sentiment}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict/file/")
async def predict_file_endpoint(file: UploadFile = File(...)):
    try:
        results = predict_file(file)
        return {"results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
