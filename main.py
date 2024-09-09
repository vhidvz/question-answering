import os
import dotenv
import uvicorn

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from model import LanguageIdentification

# Load environment variables from .env file
dotenv.load_dotenv()

app = FastAPI(
    title="Language Identification",
    description="https://fasttext.cc/docs/en/language-identification.html",
    version="1.0.0"
)

model = LanguageIdentification().load_model()


class RequestModel(BaseModel):
    text: str


class ResponseModel(BaseModel):
    language: str
    confidence: float


@app.post("/detect", response_model=ResponseModel)
def detect(payload: RequestModel):
    # Ensure text is provided
    if not payload.text:
        raise HTTPException(status_code=400, detail="Text input is required.")

    # Detect language
    try:
        return model.detect(payload.text)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", '8000'))

    print("ReDoc OpenAPI: http://0.0.0.0:{}/redoc".format(port))
    print("Swagger OpenAPI: http://0.0.0.0:{}/docs".format(port))

    uvicorn.run(app, host="0.0.0.0", port=port)
