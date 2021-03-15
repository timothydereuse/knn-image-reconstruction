import numpy as np
from matplotlib import pyplot as plt
from skimage.io import imread, imsave
from skimage.segmentation import felzenszwalb, mark_boundaries
from skimage.feature import match_template
from GIST import GIST
from make_dataset import build_gist_db
from sklearn.neighbors import NearestNeighbors

tgt_img_path = r"C:\Users\Tim\Documents\Shared\ghlka\ffxiv_04042015_161929.png"
db_file = 'test.hdf5'

img = imread(tgt_img_path, 0)
segments = felzenszwalb(img, scale=70.0, sigma=0.95, min_size=10)
gist_db, gist_db_idxs = build_gist_db(db_file)
nn = NearestNeighbors(n_neighbors=5, radius=0.4)
nn.fit(gist_db)



gist = GIST()
imsave('test.png', mark_boundaries(img, segments))

idx = 300

mask = (segments == idx)

hz_projection = np.sum(mask, 0)
vt_projection = np.sum(mask, 1)
ul = (hz_projection.nonzero()[0][0], vt_projection.nonzero()[0][0])
lr = (hz_projection.nonzero()[0][-1], vt_projection.nonzero()[0][-1])
height = lr[0] - ul[0]
width = lr[1] - ul[1]
if height > width:
    expand_amt = (height - width) // 2
    odd_adjust = (height - width) % 2 == 1
    new_ul = [ul[0], ul[1] - expand_amt - odd_adjust]
    new_lr = [lr[0], lr[1] + expand_amt]
elif width > height:
    expand_amt = (width - height) // 2
    odd_adjust = (width - height) % 2 == 1
    new_ul = [ul[0] - expand_amt - odd_adjust, ul[1]]
    new_lr = [lr[0] + expand_amt, lr[1]]
else:
    new_ul, new_lr = ul, lr

for x in [0, 1]:
    if new_ul[x] < 0:
        new_lr[x] += (0 - new_ul[x])
        new_ul[x] = 0
    if new_lr[x] >= img.shape[x]:
        new_ul[x] += (new_lr[x] - img.shape[x])
        new_ul[x] = img.shape[x]

neighborhood = img[new_ul[0]:new_lr[0], new_ul[1]:new_lr[1]].mean(2).round()
descriptor = gist._gist_extract(neighborhood)

nn.kneighbors(descriptor.reshape(1, -1))
