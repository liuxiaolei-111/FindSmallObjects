import matplotlib as mpl
# we cannot use remote server's GUI, so set this
mpl.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm as CM
from PIL import Image
import h5py
import numpy as np
import cv2

img_path = "/home/datamining/Datasets/CrowdCounting/shanghai/part_A_final/test_data/images/IMG_54.jpg"
img = cv2.imread(img_path)
# adaptive gaussian filter
adaptive = h5py.File(img_path.replace('.jpg','.h5').replace('images','ground_truth'),'r')
adaptive = np.asarray(adaptive['density'])
heatmap = adaptive/np.max(adaptive)
# must convert to type unit8
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img1 = heatmap*0.9+img
# fixed gaussian filter
fixed = h5py.File(img_path.replace('shanghai','shanghaitech_fixed').replace('.jpg','.h5').replace('images','ground_truth'),'r')
fixed = np.asarray(fixed['density'])
heatmap = fixed/np.max(fixed)
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
superimposed_img2 = heatmap*0.9+img
h = superimposed_img2.sha
slip = np.asarray()
imgs = np.hstack([img, superimposed_img1,superimposed_img2])

cv2.imwrite('figs/superimposed_img.jpg', imgs)