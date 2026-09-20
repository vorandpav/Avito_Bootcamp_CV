import io
import random

from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as F

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


class RandomNormalRotation:
    def __init__(
            self,
            std: float,
            limit: float,
            probability: float,
    ):
        self.std = std
        self.limit = limit
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        angle = random.gauss(0.0, self.std)

        angle = max(
            -self.limit,
            min(self.limit, angle),
        )

        return F.rotate(
            image,
            angle=angle,
            interpolation=transforms.InterpolationMode.BILINEAR,
            expand=False,
        )


class RandomGaussianNoise:
    def __init__(
            self,
            std: float,
            probability: float,
    ):
        self.std = std
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        noise = image.new_empty(image.shape).normal_(
            mean=0.0,
            std=self.std,
        )

        return (image + noise).clamp(0.0, 1.0)


class RandomLightBlur:
    def __init__(self, probability: float):
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        return image.filter(
            ImageFilter.GaussianBlur(radius=0.5)
        )


class RandomJPEGCompression:
    def __init__(
            self,
            quality_min: int,
            quality_max: int,
            probability: float,
    ):
        self.quality_min = quality_min
        self.quality_max = quality_max
        self.probability = probability

    def __call__(self, image):
        if random.random() > self.probability:
            return image

        quality = random.randint(
            self.quality_min,
            self.quality_max,
        )

        buffer = io.BytesIO()

        image.save(
            buffer,
            format="JPEG",
            quality=quality,
        )

        buffer.seek(0)

        with Image.open(buffer) as compressed:
            return compressed.convert("RGB").copy()


def create_transform(
        input_height: int,
        input_width: int,
        augmentation_config=None,
):
    transform_list: list = [
        transforms.Resize(
            (input_height, input_width)
        )
    ]

    if (
            augmentation_config is not None
            and augmentation_config.enabled
    ):
        transform_list.extend(
            [
                RandomNormalRotation(
                    std=augmentation_config.rotation_std,
                    limit=augmentation_config.rotation_limit,
                    probability=augmentation_config.rotation_probability,
                ),

                transforms.RandomApply(
                    [
                        transforms.ColorJitter(
                            brightness=augmentation_config.brightness,
                            contrast=augmentation_config.contrast,
                            saturation=augmentation_config.saturation,
                            hue=augmentation_config.hue,
                        )
                    ],
                    p=augmentation_config.color_jitter_probability,
                ),

                transforms.RandomPerspective(
                    distortion_scale=(
                        augmentation_config.perspective_distortion_scale
                    ),
                    p=augmentation_config.perspective_probability,
                ),

                RandomLightBlur(
                    probability=augmentation_config.blur_probability,
                ),

                RandomJPEGCompression(
                    quality_min=(
                        augmentation_config.compression_quality_min
                    ),
                    quality_max=(
                        augmentation_config.compression_quality_max
                    ),
                    probability=(
                        augmentation_config.compression_probability
                    ),
                ),
            ]
        )

    transform_list.extend(
        [
            transforms.ToTensor(),

            RandomGaussianNoise(
                std=augmentation_config.noise_std
                if augmentation_config is not None
                else 0.0,
                probability=augmentation_config.noise_probability
                if augmentation_config is not None
                else 0.0,
            ),

            transforms.Normalize(
                mean=MEAN,
                std=STD,
            ),
        ]
    )

    return transforms.Compose(transform_list)


def create_inference_transform(
        input_height: int,
        input_width: int,
):
    """
    Реальный inference: никаких augmentations.
    """

    return transforms.Compose(
        [
            transforms.Resize(
                (input_height, input_width)
            ),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=MEAN,
                std=STD,
            ),
        ]
    )
