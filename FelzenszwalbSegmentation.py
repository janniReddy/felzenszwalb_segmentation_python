import numpy as np
import cv2
from Graph import Graph

imgFile = './data/shore.ppm'
# tempFile = './data/temp.ppm'
bgr_im = cv2.imread(imgFile)
# cv2.imwrite(tempFile, bgr_im)
bgr_im = cv2.GaussianBlur(bgr_im, (11, 11), 0)
graph = Graph(bgr_im.shape[0], bgr_im.shape[1])
k = 500
min_size = 20
bgr_im = graph.segment_image(bgr_im, k, min_size)
cv2.imwrite('./data/shore_output.ppm', bgr_im)