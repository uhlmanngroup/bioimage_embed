import albumentations as A
import cv2

DEFAULT_AUGMENTATION = A.Compose(
    [
        # Flip the images horizontally or vertically with a 50% chance
        A.OneOf(
            [
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
            ],
            p=0.5,
        ),
        # Rotate the images by a random angle within a specified range
        A.Rotate(limit=45, p=0.5),
        # Randomly scale the image intensity to adjust brightness and contrast
        A.RandomGamma(gamma_limit=(80, 120), p=0.5),
        # Apply random elastic transformations to the images
        A.ElasticTransform(
            alpha=1,
            sigma=50,
            alpha_affine=50,
            p=0.5,
        ),
        # Shift the image channels along the intensity axis
        A.ChannelShuffle(p=0.5),
        # Add a small amount of noise to the images
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.5),
        # Crop a random part of the image and resize it back to the original size
        A.RandomResizedCrop(
            height=512, width=512, scale=(0.9, 1.0), ratio=(0.9, 1.1), p=0.5
        ),
        # Adjust image intensity with a specified range for individual channels
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
    ]
)
