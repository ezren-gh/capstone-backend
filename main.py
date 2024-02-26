from contextlib import asynccontextmanager
from typing import Union
import io
import numpy as np

from fastapi import FastAPI, UploadFile, Response, Depends, File
from pydantic import BaseModel
from PIL import Image
from models import vision_models

models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    models["base"] = vision_models.base_glip_inference
    models["glip"] = vision_models.glip_inference
    models["fiber"] = vision_models.fiber_inference

    yield

    models.clear()

app = FastAPI(lifespan=lifespan)

class DemoRequest(BaseModel):
    text: str
    ground_tokens: str = ""

@app.get("/")
def read_root():
    return {"Hello": "World"}

@app.post("/detection/base")
async def desco_inference(body: DemoRequest = Depends(), image: UploadFile = File(...)):
    raw = await image.read()
    img = np.array(Image.open(io.BytesIO(raw)))
    res = models["base"](img, body.text)

    im = Image.fromarray(res)
    
    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format='PNG')
        im_bytes = buf.getvalue()
    
    return Response(content=im_bytes, media_type='image/png')

@app.post("/detection/glip")
async def glip_inference(body: DemoRequest = Depends(), image: UploadFile = File(...)):
    raw = await image.read()
    img = np.array(Image.open(io.BytesIO(raw)))
    res = models["glip"](img, body.text, body.ground_tokens)
    im = Image.fromarray(res)
    
    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format='PNG')
        im_bytes = buf.getvalue()
    
    return Response(content=im_bytes, media_type='image/png')

@app.post("/detection/fiber")
async def desco_inference(body: DemoRequest = Depends(), image: UploadFile = File(...)):
    raw = await image.read()
    img = np.array(Image.open(io.BytesIO(raw)))
    res = models["fiber"](img, body.text,  body.ground_tokens)
    im = Image.fromarray(res)
    
    # save image to an in-memory bytes buffer
    with io.BytesIO() as buf:
        im.save(buf, format='PNG')
        im_bytes = buf.getvalue()
    
    return Response(content=im_bytes, media_type='image/png')
