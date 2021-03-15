
import skimage.io as io
import skimage.transform as tf
import os
import numpy as np
from skimage.util import img_as_ubyte, img_as_uint, img_as_bool
from skimage.color import gray2rgb, rgba2rgb, rgb2gray
from skimage.io import imread, imshow, imsave, ImageCollection
from skimage.transform import rescale, resize, downscale_local_mean
from skimage.filters import threshold_local, threshold_otsu, try_all_threshold
from skimage.morphology import skeletonize, binary_closing, binary_opening, binary_erosion, thin, octagon, area_closing, diameter_closing
from skimage.draw import rectangle, rectangle_perimeter
from augment_images import run_augs
from itertools import groupby
from GIST import GIST
import h5py

def build_gist_db(hdf5_fname):
    gists = []
    names = []
    with h5py.File(hdf5_fname, 'a') as f:
        keys = list(f.keys())
        grp = f["gist"]
        for key in grp.keys():
            gists.append(grp[key][:])
            names += [(key, i) for i in range(grp[key].shape[1])]
    arr = np.concatenate(gists, 1).swapaxes(0, 1)
    return arr, names

def get_fnames(base_dir):
    file_types = ['jpeg', 'png', 'jpg']
    images = []
    for dir, folders, files in os.walk(base_dir):
        if dir.split('\\')[-1] in ['real', 'ffx', 'collect_result', 'slime']:
            continue
        for f in files:
            if (f.split('.')[-1] in file_types):
                images.append(os.path.join(dir, f))
    return images

def crop_to_center(img):
    if img.shape[0] > img.shape[1]:
        left = (img.shape[0] - img.shape[1]) // 2
        right = (img.shape[0] + img.shape[1]) // 2
        vary = int(np.random.uniform(-1, 1) * (img.shape[0] - img.shape[1]) / 4)
        img = img[left + vary:right + vary, :]
    else:
        top = (img.shape[1] - img.shape[0]) // 2
        bottom = (img.shape[1] + img.shape[0]) // 2
        vary = int(np.random.uniform(-1, 1) * (img.shape[1] - img.shape[0]) / 4)
        img = img[:, top + vary:bottom + vary]
    return img


def threshold_img(img, block_size=17, offset=0.01):
    img_bw = rgb2gray(img)
    local_thresh = threshold_local(img_bw, block_size, offset=offset)
    img_bin = img_bw > local_thresh
    # img_clos = area_closing(img_as_ubyte(img_bin), 36) > 0
    return img_bin

def make_examples(out_file, aug_factor=10, target_size=(512, 512), resize_under=1024.0, dset_size=8):

    img_list = []

    def dump_imgs(img_list, dset_name):
        print(f'    dumping to {dset_name}...')
        arr = np.stack(img_list, 0)
        with h5py.File(out_file, 'a') as f:
            dset = f.create_dataset(dset_name, data=arr, dtype='uint8', chunks=True, compression="gzip")
        img_list = []

    all_fnames = get_fnames(r'C:\Users\Tim\Documents\Shared\ghlka\TumblThree\Blogs\obtain')[:20]

    for key, group in groupby(all_fnames, lambda x: x.split('\\')[-2]):
        print(f'processing {key}...')
        for i, fname in enumerate(list(group)):

            print(f"    processing {i}: {fname}...")
            img = imread(fname)

            # REMOVE TRANSPARENCY IF PRESENT
            if len(img.shape) < 3:
                img = gray2rgb(img)
            elif img.shape[2] > 3:
                img = rgba2rgb(img)

            if min(img.shape[:2]) > resize_under:
                ratio = 512.0 / min(img.shape[:2])
                new_shape = [int(img.shape[0] * ratio), int(img.shape[1] * ratio)]
                new_img = (resize(img / 255, new_shape) * 255).astype('uint8')

            # PRELIMINARY THRESHOLD FOR CROPPING
            img_bin = threshold_img(img)
            # CROP OUT WHITESPACE ON EDGES
            [rows, columns] = np.where(1 - img_bin)
            img_crop = img[min(rows):max(rows), min(columns):max(columns)]

            # RUN AUGMENTATIONS
            stack = np.array(
                [img_crop for _ in range(aug_factor)],
                dtype=np.uint8
            )

            stack = run_augs(stack)
            img_list += stack

            if i > 0 and (i % dset_size == 0):
                dump_imgs(img_list, f'{key}_{i}')
                img_list = []

        # dump last part
        if len(img_list) > 0:
            dump_imgs(img_list, f'{key}_{i}')


def make_gist_descriptors(out_file):
    gist = GIST()

    with h5py.File(out_file, 'a') as f:
        keys = list(f.keys())
        grp = f.create_group("gist")

    for k in keys:
        if k == 'gist':
            continue
        print(f'starting {k}...')
        with h5py.File(out_file, 'a') as f:
            imgs = f[k][:]
            imgs_gray = np.round(np.mean(imgs, 3))

            gists = []
            for i in range(imgs_gray.shape[0]):
                gists.append(gist._gist_extract(imgs_gray[i]))
                if not i % 20:
                    print(f'{i} of {imgs_gray.shape[0]} in {k}...')

            gists = np.stack(gists, 1)

            grp = f['gist']
            dset = grp.create_dataset(k, data=gists, chunks=True, compression="gzip")

if __name__ == '__main__':
    make_examples('test.hdf5')
    make_gist_descriptors('test.hdf5')
