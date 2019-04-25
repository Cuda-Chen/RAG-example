from skimage import data, io, segmentation, color
from skimage.measure import regionprops
from skimage.future import graph
import numpy as np
import argparse
import os

def show_image(image):
    io.imshow(image)
    io.show()

parser = argparse.ArgumentParser()
parser.add_argument('input_image', type=str, help='input image path')
parser.add_argument('num_superpixel', type=int, help='number of segments')
parser.add_argument('compactness', type=int, help='compactness param of SLIC')
parser.add_argument('thresh', type=float, help='threshold of combining edge')
args = parser.parse_args()

img = io.imread(args.input_image)
outputfile = os.path.splitext(args.input_image)[0] \
+ '_' + str(args.num_superpixel) \
+ '_' + str(args.compactness) \
+ '_' + str(args.thresh) + '.bmp'
#show_image(img)

labels = segmentation.slic(img, n_segments=args.num_superpixel, compactness=args.compactness)
labels = labels + 1 # so that regionprops will not to ignore no labelled region
regions = regionprops(labels)

label_rgb = color.label2rgb(labels, img, kind='avg')
#show_image(label_rgb)

# uncomment these two lines if you want the SLIC output with edge boundary
#label_rgb = segmentation.mark_boundaries(label_rgb, labels, (0, 0, 0))
#show_image(label_rgb)

rag = graph.rag_mean_color(img, labels)

labels1 = graph.cut_threshold(labels, rag, args.thresh)
out = color.label2rgb(labels1, img, kind='avg')
#show_image(out)

io.imsave(outputfile, out)
