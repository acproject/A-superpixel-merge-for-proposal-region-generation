from skimage import data, io, segmentation, color
from skimage.future import graph
from skimage.measure import regionprops
import numpy as np
import cv2;
from  PIL import Image;
from matplotlib import pyplot as plt, colors


def _weight_mean_color(graph, src, dst, n):
    ''':cvar
    Callback to handle merging nodes by recomputing mean color.
    The method expects that the mean color of `dst` is already computed.

    :parameters
    -------
    graph : RAG(Region Adjacency Graph)
        the graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbot of `src` or `dst` or both.
    :return
    data: dict
        A dictionary with the `weight` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    '''

    diff = graph.nodes[dst]['mean color'] - graph.nodes[n]['mean color']
    diff = np.linalg.norm(diff)
    return {'weight': diff}

def merge_mean_color(graph, src ,dst):
    ''':cvar
    Callback called before merging two nodes of a mean color distance graph.
    This method computes the mean color of 'dst'.
    :parameter
    -----------
    graph: RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    '''

    graph.nodes[dst]['total color'] += graph.nodes[src]['total color']
    graph.nodes[dst]['pixel count'] += graph.nodes[src]['pixel count']
    graph.nodes[dst]['mean color'] = (graph.nodes[dst]['total color'] / graph.nodes[dst]['pixel count'])

img = Image.open('../images/12.png')
img = img.convert("RGB")

img = np.array(img)
# print(img)
# print("img:", img.shape)
# print("img:", type(img))


labels = segmentation.slic(img, compactness=10, n_segments=1200)
g = graph.rag_mean_color(img, labels, mode='distance')
cmap = colors.ListedColormap(['#6599FF', '#ff9900'])
# graph.show_rag(labels, g, img, img_cmap=cmap)

labels2 = graph.merge_hierarchical(labels, g, thresh=30, rag_copy=False,
                                   in_place_merge=True, merge_func=merge_mean_color,
                                   weight_func=_weight_mean_color)
out = color.label2rgb(labels2, img, kind='overlay', bg_label=255)
# out2 = color.label2rgb(labels2, img, kind='overlay', bg_label=2)
out = segmentation.mark_boundaries(out, labels2, (0, 0, 0))

io.imshow(out)
io.show()
