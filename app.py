# Reference: https://huggingface.co/spaces/haotiz/glip-zeroshot-demo/blob/main/app.py 

import requests
import os
from io import BytesIO
from PIL import Image
import numpy as np
from pathlib import Path
import gradio as gr
from urllib.parse import urljoin

URL = "http://127.0.0.1:8000"
BASE_GLIP_URL = urljoin(URL, "/detection/base")
DESCO_GLIP_URL = urljoin(URL, "/detection/glip")
DESCO_FIBER_URL = urljoin(URL, "/detection/fiber")

def predict(image, text, ground_tokens=""):
    params = {"text": text, "ground_tokens": ground_tokens}
    files = {"image": ("inf.png", image)}

    glip_res = requests.post(DESCO_GLIP_URL, params=params, files=files)
    fiber_res = requests.post(DESCO_FIBER_URL, params=params, files=files)
    base_res = requests.post(BASE_GLIP_URL, params=params, files=files)

    return glip_res, fiber_res, base_res

image = gr.inputs.Image()

gr.Interface(
    description="Object Recognition with DesCo (https://github.com/liunian-harold-li/DesCo)",
    fn=predict,
    inputs=["image", "text", "text"],
    outputs=[
        gr.outputs.Image(
            type="pil",
            label="DesCo-GLIP"
        ),
        gr.outputs.Image(
            type="pil",
            label="DesCo-FIBER"
        ),
        gr.outputs.Image(
            type="pil",
            label="GLIPv2"
        )
    ],
    examples=[
        ["./1.jpg", "A clown making a balloon animal for a pretty lady.", "clown"],
        ["./1.jpg", "A clown kicking a soccer ball for a pretty lady.", "clown"],
        ["./2.jpg", "A kind of tool, wooden handle with a round head.", "tool"],
        ["./3.jpg", "Bumblebee, yellow with black accents.", "Bumblebee"],
    ]
).launch()
