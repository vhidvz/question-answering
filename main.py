import os
import dotenv
import uvicorn

from pydantic import BaseModel
from fastapi import FastAPI, HTTPException

from model import QuestionAnswering


# Load environment variables from .env file
dotenv.load_dotenv()

app = FastAPI(
    title="Question Answering",
    description="https://huggingface.co/m3hrdadfi/gpt2-persian-qa",
    version="1.0.0"
)

model = QuestionAnswering().load_model()


class RequestModel(BaseModel):
    lang: str
    query: str


class ResponseModel(BaseModel):
    answer: str


@app.post("/answer", response_model=ResponseModel)
def detect(payload: RequestModel):
    # Ensure text is provided
    if not payload.query or not payload.lang:
        raise HTTPException(status_code=400, detail="Text input is required.")

    # Answer the question
    try:
        return model.answer(payload.query, payload.lang)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class DocumentsModel(BaseModel):
    contents: list[str]


@app.post("/documents")
def detect(payload: DocumentsModel):
    # Ensure text is provided
    if not len(payload.contents):
        raise HTTPException(status_code=400, detail="Text input is required.")

    # Add a new document
    try:
        return model.add_documents(payload.contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", '8000'))

    print("ReDoc OpenAPI: http://0.0.0.0:{}/redoc".format(port))
    print("Swagger OpenAPI: http://0.0.0.0:{}/docs".format(port))

    uvicorn.run(app, host="0.0.0.0", port=port)
