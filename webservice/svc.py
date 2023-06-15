from fastapi import FastAPI, File, Form, UploadFile
import numpy as np
import uvicorn
import os
import cv2
from PIL import Image
from io import BytesIO
import time
from fastapi.openapi.docs import get_swagger_ui_html
from fastapi.staticfiles import StaticFiles

import json
from pydantic import BaseModel
from typing import Optional
from auomotiondetection.backgroundsubtractionmovingaverage.core import BackgroundSubtractMovingAverage

app = FastAPI()

#############################################################################
print(os.getcwd())
app.mount("/static", StaticFiles(directory="auomotiondetection/webservice/static"), name="static")

PoolMotion = {}


@app.get("/docs2", include_in_schema=False)
async def custom_swagger_ui_html():
    """
    For local js, css swagger in AUO
    :return:
    """
    return get_swagger_ui_html(
        openapi_url=app.openapi_url,
        title=app.title + " - Swagger UI",
        oauth2_redirect_url=app.swagger_ui_oauth2_redirect_url,
        swagger_js_url="/static/swagger-ui-bundle.js",
        swagger_css_url="/static/swagger-ui.css",
    )


##############################################################################

@app.get("/")
def HelloWorld():
    return {"Hello": "World"}


class StructureBase(BaseModel):
    uuid: str
    minArea: Optional[int] = 2500
    updateWeight: Optional[float] = 0.2
    if_union: Optional[bool] = False

    # 以下兩個 functions 請盡可能不要更動-----------------------
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value


@app.post("/MotionDetection/")
def MotionDetection(data: StructureBase = Form(...), file: UploadFile = File(...)):
    t0 = time.time()
    # get image
    cv2_img = bytes_to_cv2image(file.file.read())
    # input_dict = data
    uuid = data.uuid
    minArea = data.minArea
    updateWeight = data.updateWeight
    if_union = data.if_union

    if uuid in PoolMotion.keys():
        motion = PoolMotion[uuid]
    else:
        motion = BackgroundSubtractMovingAverage(minArea, updateWeight)
        PoolMotion[uuid] = motion
    xywhs, cnts = motion.update(cv2_img)

    # combine multiple motion box to one
    if xywhs and if_union:
        ls_x_min = []
        ls_x_max = []
        ls_y_min = []
        ls_y_max = []
        for item in xywhs:
            ls_x_min.append(item['x'])
            ls_x_max.append(item['x'] + item['w'])
            ls_y_min.append(item['y'])
            ls_y_max.append(item['y'] + item['h'])
        x = min(ls_x_min)
        y = min(ls_y_min)
        w = max(ls_x_max) - x
        h = max(ls_y_max) - y
        xywhs = [{"x": x, "y": y, "w": w, "h": h}]

    # tmp_cnts = [item.tolist() for item in cnts]
    # output_dict = {"xywhs": xywhs, "cnts": tmp_cnts}
    t1 = time.time()
    fps = 1.0 / (t1 - t0)
    output_dict = {"xywhs": xywhs, 'fps': fps}
    return output_dict


def bytes_to_cv2image(imgdata):
    cv2img = cv2.cvtColor(np.array(Image.open(BytesIO(imgdata))), cv2.COLOR_RGB2BGR)
    return cv2img


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5111))
    uvicorn.run(app, log_level='info', host='0.0.0.0', port=port)
