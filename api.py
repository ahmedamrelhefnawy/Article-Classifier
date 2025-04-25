from fastapi.params import Body
from classifier import pipeline, CURRENT_DIR
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

app = FastAPI()
app.mount(CURRENT_DIR+"/static" , StaticFiles(directory="static"), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipe = pipeline()

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)


class PredictionInput(BaseModel):
    text: str

class PredictionOutput(BaseModel):
    prediction: str

@app.post("/predict", response_model=PredictionOutput)
async def predict(input: PredictionInput) -> PredictionOutput:
    if input.text:
        result = pipe.predict(input.text)
        return PredictionOutput(prediction=result)
    
    return PredictionOutput(prediction="No Text Provided")

if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)