import torch
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    # Reduce large images in stages before the final bicubic resize.
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    # Resize the short side to image_size while preserving the aspect ratio.
    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    # Extract a centered square from the remaining long side.
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


class RandomCropWithScale:
    """Expand by a sampled scale, then take a random square crop."""
    def __init__(self, image_size, scales):
        self.image_size = int(image_size)
        self.scales = tuple(float(scale) for scale in scales)

    def __call__(self, image):
        scale_index = torch.randint(len(self.scales), size=(1,)).item()
        expanded_size = max(
            self.image_size,
            int(round(self.image_size * self.scales[scale_index])),
        )
        # Prepare a larger square, then sample an image_size crop within it.
        image = center_crop_arr(image, expanded_size)
        return transforms.RandomCrop(self.image_size)(image)


def build_train_transform(img_size, random_crop_scales=None):
    if random_crop_scales:
        crop = RandomCropWithScale(img_size, random_crop_scales)
    else:
        crop = transforms.Lambda(lambda img: center_crop_arr(img, img_size))
    return transforms.Compose([
        crop,
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.PILToTensor(),
    ])
