import albumentations as A

transform_fn = A.Compose([
    A.OneOf([
        A.IAAAdditiveGaussianNoise(),
        A.GaussNoise(),
    ], p=.2),
    A.OneOf([
        A.MotionBlur(p=.2),
        A.MedianBlur(blur_limit=3, p=.2),
        A.Blur(blur_limit=3, p=.2),
    ], p=.2),
    A.OneOf([
        A.CLAHE(clip_limit=2),
        A.IAASharpen(),
        A.IAAEmboss(),
        A.RandomBrightnessContrast(),
    ], p=0.2),
    A.OneOf([
        A.ChannelShuffle(),
        A.HueSaturationValue(),
        A.RGBShift(),
        A.ToSepia(),
    ], p=0.2),
    A.OneOf([
        A.RandomSunFlare(flare_roi=(0, 0, 1., 1.), src_radius=100, p=.1),
        A.RandomFog(fog_coef_lower=0.2, fog_coef_upper=0.3, p=.1),
        A.RandomShadow(p=.1),
        A.RandomRain(p=.1, blur_value=1),
    ], p=0.1)
])