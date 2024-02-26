import os
import warnings
import math

warnings.filterwarnings("ignore")

os.system("python setup.py build develop --user")

from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo
from copy import deepcopy

params = {
    "skip_name": False,
    "override_color": (175, 225, 175),
    "text_size": 0.8,
    "text_pixel": 2,
    "box_alpha": 1.0,
    "box_pixel": 3,
    "text_offset_original": 8,
}

def prepare_model(cfg, config_file, weight_file):
    cfg.local_rank = 0
    cfg.num_gpus = 1
    cfg.merge_from_file(config_file)
    cfg.merge_from_list(["MODEL.WEIGHT", weight_file])
    cfg.merge_from_list(["MODEL.DEVICE", "cuda"])

    return GLIPDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        show_mask_heatmaps=False
    )

glip_demo = prepare_model(deepcopy(cfg), "./models/config/desco_glip.yaml", "./models/weights/desco_glip_tiny.pth")
fiber_demo = prepare_model(deepcopy(cfg), "./models/config/desco_fiber.yaml", "./models/weights/desco_fiber_base.pth")
base_glip_demo = prepare_model(deepcopy(cfg), "./models/config/configs_pretrain_glip_Swin_T_O365_GoldG.yaml", "./models/weights/glip_tiny_model_o365_goldg_cc_sbu.pth")

def glip_inference(image, text, ground_tokens=""):
    img_len = min(image.shape[:2])
    params["text_size"] = math.ceil(img_len/1000)
    params["text_pixel"] = math.ceil(img_len/1000*3)
    ground_tokens = None if ground_tokens.strip() == "" else ground_tokens.strip().split(";")
    result, _ = glip_demo.run_on_web_image(deepcopy(image[:, :, [2, 1, 0]]), text, 0.5, ground_tokens, **params)
    return result[:, :, [2, 1, 0]]


def fiber_inference(image, text, ground_tokens=""):
    img_len = min(image.shape[:2])
    params["text_size"] = math.ceil(img_len/1000)
    params["text_pixel"] = math.ceil(img_len/1000*3)
    ground_tokens = None if ground_tokens.strip() == "" else ground_tokens.strip().split(";")
    result, _ = fiber_demo.run_on_web_image(deepcopy(image[:, :, [2, 1, 0]]), text, 0.5, ground_tokens, **params)
    return result[:, :, [2, 1, 0]]

def base_glip_inference(image, text):
    img_len = min(image.shape[:2])
    params["text_size"] = math.ceil(img_len/1000)
    params["text_pixel"] = math.ceil(img_len/1000*3)
    result, _ = glip_demo.run_on_web_image(image[:, :, [2, 1, 0]], text, 0.5, **params)
    return result[:, :, [2, 1, 0]]
