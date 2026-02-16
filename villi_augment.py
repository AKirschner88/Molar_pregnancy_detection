from albumentations.core.transforms_interface import ImageOnlyTransform
import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
from skimage.color import rgb2hed, hed2rgb
import numbers
import cv2

class HEDJitterAugmentation(ImageOnlyTransform):
    def __init__(self,alpha,beta,always_apply=False,p=0.5):
        super(HEDJitterAugmentation, self).__init__(always_apply, p)
        if isinstance(alpha,numbers.Number):
            self.alpha = (-alpha,alpha)
        elif isinstance(alpha,tuple):
            if alpha[0] <= alpha[1]:
                self.alpha = alpha
            else:
                raise ValueError
        else:
            raise ValueError
        if isinstance(beta,numbers.Number):
            self.beta = (-beta,beta)
        elif isinstance(beta,tuple):
            if beta[0] <= beta[1]:
                self.beta = beta
            else:
                raise ValueError
        else:
            raise ValueError
        self.cap = np.array([1.87798274, 1.13473037, 1.57358807])

    def adjust_HED(self,img):
        img = np.array(img)
        
        alpha = np.random.uniform(1+self.alpha[0], 1+self.alpha[1], (1, 3))
        betti = np.random.uniform(self.beta[0], self.beta[1], (1, 3))

        alpha[0,2]=min(alpha[0,2],2*alpha[0,:2].prod()/alpha[0,:2].sum())

        s = rgb2hed(img)/self.cap
        s = alpha * (s + betti)
        nimg = hed2rgb(s*self.cap)

        return (255*nimg).clip(0,255).astype(np.uint8)

    def apply(self, image, **params):
        # Apply your custom color augmentation function to the image
        augmented_image = self.adjust_HED(image)

        # Return the augmented image and the unchanged mask
        return augmented_image


def randaugment():
    p = 0.8
    size = 1024  # typically this is the size of the input image (was 1024 in hoverfast)

    aug_always = [HEDJitterAugmentation((-0.4, 0.4), (-0.005, 0.01), p=p),
                  A.VerticalFlip(p=p),
                  A.HorizontalFlip(p=p),
                  A.RandomResizedCrop(size, size, scale=(0.95, 1.05), always_apply=True, p=p),
                  A.Rotate(p=1, value=(255, 255, 255), crop_border=True),
                  A.RandomSizedCrop((size//2, size//2), size//2, size//2)
                  ]

    aug_intensity = [A.RandomBrightnessContrast(p=p, brightness_limit=0.2, contrast_limit=0.2),
                     A.RandomGamma(p=p, gamma_limit=(65, 140), eps=1e-7)
                     ]

    blur = [A.Blur(blur_limit=5, p=0.3),
            A.MotionBlur(p=0.3, blur_limit=(3,5))
            ]

    aug_noise = [A.GaussNoise(p=p, var_limit=(120,600)), 
                 A.ISONoise(p=p, intensity=(0.1,0.4), color_shift=(0.05,0.2)),
                 A.MultiplicativeNoise(p=p, multiplier=(0.75, 1.25), elementwise=True)
                 ]

    noise_ops = np.random.choice(aug_noise, 1, replace=False).tolist()
    intensity_ops = np.random.choice(aug_intensity, 1, replace=False).tolist()
    blur_ops = np.random.choice(blur, 1, replace=False).tolist()
    transforms = A.Compose(aug_always + intensity_ops + blur_ops + noise_ops)
    
    return transforms


def noaugment():
    return A.NoOp()


def randaugment_unet():
    p = 0.5
    size = 1024

    transforms = A.Compose([
        A.VerticalFlip(p=p),
        A.HorizontalFlip(p=p),
        A.HueSaturationValue(hue_shift_limit=(-25, 0), sat_shift_limit=0, val_shift_limit=0, p=1),
        A.Rotate(p=1, border_mode=cv2.BORDER_REFLECT_101),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        # ElasticTransform(always_apply=True, approximate=True, alpha=150, sigma=8, alpha_affine=50),
        A.RandomSizedCrop(min_max_height=(size, size), size=(size, size), height=size, width=size, p=1),
        # ToTensorV2()
    ])

    return transforms


def randaugment_rmaskcnn():
    p = 0.5
    size = 1024

    transforms = A.Compose([
        A.VerticalFlip(p=p),
        A.HorizontalFlip(p=p),
        A.HueSaturationValue(hue_shift_limit=(-25, 0), sat_shift_limit=0, val_shift_limit=0, p=1),
        A.Rotate(
            p=1,
            border_mode=cv2.BORDER_REFLECT_101,
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
        ),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        A.RandomSizedCrop(
            min_max_height=(size, size),
            size=(size, size),
            interpolation=cv2.INTER_LINEAR,
            mask_interpolation=cv2.INTER_NEAREST,
            p=1
        ),
    ])

    return transforms



def randaugment_hoverfast_villi():
    p = 0.8
    size = 1024

    aug_always = [HEDJitterAugmentation((-0.4, 0.4), (-0.005, 0.01), p=p),
                  A.VerticalFlip(p=p),
                  A.HorizontalFlip(p=p),
                  A.RandomResizedCrop(size, size, scale=(0.95, 1.05), always_apply=True, p=p),
                  A.Rotate(p=1, border_mode=cv2.BORDER_REFLECT_101),
                  ]

    aug_intensity = [A.RandomBrightnessContrast(p=p, brightness_limit=0.2, contrast_limit=0.2),
                     A.RandomGamma(p=p, gamma_limit=(65, 140), eps=1e-7)
                     ]

    blur = [A.Blur(blur_limit=5, p=0.3),
            A.MotionBlur(p=0.3, blur_limit=(3, 5))
            ]

    aug_noise = [A.GaussNoise(p=p, var_limit=(120, 600)),
                 A.ISONoise(p=p, intensity=(0.1, 0.4), color_shift=(0.05, 0.2)),
                 A.MultiplicativeNoise(p=p, multiplier=(0.75, 1.25), elementwise=True)
                 ]

    noise_ops = np.random.choice(aug_noise, 1, replace=False).tolist()
    intensity_ops = np.random.choice(aug_intensity, 1, replace=False).tolist()
    blur_ops = np.random.choice(blur, 1, replace=False).tolist()
    transforms = A.Compose(aug_always + intensity_ops + blur_ops + noise_ops)

    return transforms