import pickle
import argparse

import matplotlib
import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg

from detectron2.utils.visualizer import Visualizer

parser = argparse.ArgumentParser()
parser.add_argument("in_path", metavar="<input path>", help="Path to texcoords_smpl_27554.pkl")
parser.add_argument("out_path", metavar="<output path>", help="Path to save UV coordinates image")
args = parser.parse_args()

f = open(args.in_path, 'rb')
data = pickle.load(f)

height = 4096
width = 4096
num_channels = 3
img = np.zeros((height, width, num_channels), dtype=np.uint8)
visualizer = Visualizer(img)

bbox_xyxy = [0.5, 0.5, 100, 100]
visualizer.draw_box(bbox_xyxy)
visualizer.output.save(args.out_path)

# densepose0 = data[0]['pred_densepose'][0]
# labels = densepose0.labels # integers
# uvs = densepose0.uv
# nonzero_label_indices = labels.nonzero(as_tuple=True)
# print(labels)
