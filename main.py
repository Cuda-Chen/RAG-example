from skimage import data, io, segmentation, color
from matplotlib import pyplot as plt
from skimage.measure import regionprops
from skimage.future import graph
import numpy as np

def show_image(image):
    width = 10.0
    height = image.shape[0] * width / image.shape[1]
    f = plt.figure(figsize=(width, height))
    plt.imshow(image)
    plt.show()

img = data.coffee()
show_image(img)

labels = segmentation.slic(img, compactness=30, n_segments=400)
labels = labels + 1 # so that regionprops will not to ignore no labelled region
regions = regionprops(labels)

label_rgb = color.label2rgb(labels, img, kind='avg')
show_image(label_rgb)

label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 0, 0))
show_image(label_rgb)

rag = graph.rag_mean_color(img, labels)

labels1 = graph.cut_threshold(labels, rag, 29)
out = color.label2rgb(labels1, img, kind='avg')
show_image(out)
