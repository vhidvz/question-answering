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
    version="1.0.3"
)

model = QuestionAnswering().load_model()


class RequestModel(BaseModel):
    query: str


class ResponseModel(BaseModel):
    answer: str


@app.post("/answer", response_model=ResponseModel)
def detect(payload: RequestModel):
    # Ensure text is provided
    if not payload.query:
        raise HTTPException(status_code=400, detail="Text input is required.")

    # Answer the question
    try:
        return model.answer(payload.query)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


class ReqDocumentsModel(BaseModel):
    contents: list[str]


class ResDocumentsModel(BaseModel):
    total: int


@app.post("/documents", response_model=ResDocumentsModel)
def detect(payload: ReqDocumentsModel):
    # Ensure text is provided
    if not len(payload.contents):
        raise HTTPException(status_code=400, detail="Text input is required.")

    # Add a new document
    try:
        return ResDocumentsModel(total=model.add_documents(payload.contents))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    port = int(os.getenv("PORT", '8000'))

    print("ReDoc OpenAPI: http://0.0.0.0:{}/redoc".format(port))
    print("Swagger OpenAPI: http://0.0.0.0:{}/docs".format(port))

    uvicorn.run(app, host="0.0.0.0", port=port)
