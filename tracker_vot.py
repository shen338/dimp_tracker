import numpy as np
import glob

import sys
import requests
import cv2
from PIL import Image
from io import BytesIO

sys.path.append('/home/dimp/tracker/maskrcnn-benchmark')
from demo.predictor import *
from maskrcnn_benchmark.config import cfg

# sys.path.append('./fastreid')
sys.path.append('/home/dimp/tracker/fast-reid')
from fastreid.config import get_cfg
from fastreid.utils.file_io import PathManager
from predictor import FeatureExtractionDemo
from fastreid.engine import DefaultPredictor

sys.path.append('/home/dimp/tracker/pytracking')
from pytracking.evaluation.dimp_tracker_vot import Tracker

import collections
Rectangle = collections.namedtuple('Rectangle', ['x', 'y', 'width', 'height'])

import vot

def setup_cfg(reid_config):
    # load config from file and command-line arguments
    cfg = get_cfg()
    # add_partialreid_config(cfg)
    cfg.merge_from_file(reid_config)
    # cfg.merge_from_list(args.opts)
    cfg.freeze()
    return cfg

detector_config = "/home/dimp/tracker//maskrcnn-benchmark/configs/caffe2/e2e_mask_rcnn_R_50_FPN_1x_caffe2.yaml"
reid_config = "/home/dimp/tracker//reid.yml"

# update the config options with the config file
cfg.merge_from_file(detector_config)

detector = COCODemo(cfg, min_image_size=200, confidence_threshold=0.4)

# def load(url):
#     """
#     Given an url of an image, downloads the image and
#     returns a PIL image
#     """
#     response = requests.get(url)
#     pil_image = Image.open(BytesIO(response.content)).convert("RGB")
#     # convert to BGR format
#     image = np.array(pil_image)[:, :, [2, 1, 0]]
#     return image
    
# image = load("http://farm3.staticflickr.com/2469/3915380994_2e611b1779_z.jpg")
# # compute predictions
# print(image.dtype)
# predictions = detector.compute_prediction(image)
# attrs = vars(predictions)
# print(dir(predictions)) 
# print(', '.join("%s: %s" % item for item in attrs.keys()))
# print(predictions.bbox)
# print(predictions.scores)
# print(predictions.get_field("scores"))
# cv2.imwrite("test.jpg", predictions)
reid_cfg = setup_cfg(reid_config)
reid_model = FeatureExtractionDemo(reid_cfg)

reid_predictor = DefaultPredictor(reid_cfg)
# original_image = original_image[:, :, ::-1]
# # Apply pre-processing to image.
# image = cv2.resize(original_image, tuple(self.cfg.INPUT.SIZE_TEST[::-1]), interpolation=cv2.INTER_CUBIC)
# # Make shape with a new batch dimension which is adapted for
# # network input
# image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
# predictions = self.predictor(image)
# image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))[None]
# embedding = reid_predictor(image)
# print(embedding.shape)

test_tracker = Tracker("dimp", "dimp18fmv")
# optional_box = [614, 389, 103, 99]
test_tracker.run_video_no_display(vot, detector, reid_predictor, "/home/dataset/dataset/LaSOT/car-14/img/", "./result")



