from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from jet.logger import logger
from typing import List, TypedDict

from span_marker import SpanMarkerModel

# Define a router instance
router = APIRouter()

# Global model and tokenizer variables
model = None
tokenizer = None


class TextRequest(BaseModel):
    text: str


class EntityResponse(BaseModel):
    span: str
    label: str
    score: float
    char_start_index: int
    char_end_index: int


class ModelLoadResponse(TypedDict):
    message: str


def load_model(model_name: str):
    global model, tokenizer
    if model is not None:
        del model
    if tokenizer is not None:
        del tokenizer

    model = SpanMarkerModel.from_pretrained(model_name)
    tokenizer = model.tokenizer

    if hasattr(tokenizer, 'pad_token_type_id') and tokenizer.pad_token_type_id is not None:
        # Resolves error in data collator trying to find missing pad_token_id
        tokenizer.pad_token_id = tokenizer.pad_token_type_id
    else:
        logger.warning(
            f"Tokenizer for model {model_name} does not have 'pad_token_type_id'")
    return model


def predict_entities(text: str, model_name: str = None) -> List[EntityResponse]:
    if model_name:
        predictor_model = load_model(model_name)
    else:
        predictor_model = model

    entities = predictor_model.predict(text)
    return [
        EntityResponse(
            span=entity['span'],
            label=entity['label'],
            score=entity['score'],
            char_start_index=entity['char_start_index'],
            char_end_index=entity['char_end_index']
        ) for entity in entities
    ]


@router.post("/predict", response_model=List[EntityResponse])
async def extract_entities(request: TextRequest) -> List[EntityResponse]:
    try:
        return predict_entities(request.text)
    except Exception as e:
        logger.error(f"Error predicting entities: {e}")
        raise HTTPException(status_code=500, detail="Internal Server Error")


@router.post("/load_model", response_model=ModelLoadResponse)
async def change_model(model_name: str) -> ModelLoadResponse:
    try:
        load_model(model_name)
        logger.info(f"Model '{model_name}' loaded successfully.")
        return {"message": f"Model '{model_name}' loaded successfully."}
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail="Failed to load model")
