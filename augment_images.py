import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa

def run_augs(inp):
    seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontal flips
        iaa.Affine(
            scale={"x": (1, 1.05), "y": (1, 1.05)},
            rotate=(-9, 9),
            shear=(-1, 1),
            mode='symmetric'
        ),
        iaa.Crop(percent=(0, 0.1)), # random crops
        iaa.CropToSquare(),
        iaa.Sometimes(
            0.2,
            iaa.GaussianBlur(sigma=(0, 0.5))
        ),
        iaa.LinearContrast((0.8, 1.2)),
        iaa.MultiplyHueAndSaturation(mul_hue=(-1, 1), mul_saturation=(0.7, 1.1)),

        iaa.Resize(512),
    ], random_order=False) # apply augmenters in random order

    return seq(images=inp)

if __name__ == "__main__":
    ia.seed(1)

    images = np.array(
        [ia.quokka(size=(64, 64)) for _ in range(32)],
        dtype=np.uint8
    )
    images_aug = seq(images=images)
