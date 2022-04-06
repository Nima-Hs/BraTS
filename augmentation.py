import os
import random
from typing import Union

import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage.interpolation import map_coordinates


class RandomHorizontalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        for i in range(indices.shape[1]):
            if self.p > np.random.uniform():
                indices[3, i] = np.flip(indices[3, i], axis=2)
        return indices


class RandomVerticalFlip:
    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        for i in range(indices.shape[1]):
            if self.p > np.random.uniform():
                indices[2, i] = np.flip(indices[2, i], axis=1)
        return indices


class RadnomHorizontalShear:
    def __init__(self, rate: Union[float, tuple]) -> None:
        if type(rate) is not tuple:
            rate = (-rate, rate)
        self.rate = rate

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        for i in range(indices.shape[1]):
            rate = np.random.uniform(low=self.rate[0], high=self.rate[1])
            indices[3, i] = rate*indices[2, i] + indices[3, i]
        return indices


class RandomVerticalShear:
    def __init__(self, rate: Union[float, tuple]) -> None:
        if type(rate) is not tuple:
            rate = (-rate, rate)
        self.rate = rate

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        for i in range(indices.shape[1]):
            rate = np.random.uniform(low=self.rate[0], high=self.rate[1])
            indices[2, i] = indices[2, i] + rate*indices[3, i]
        return indices


class RandomHorizontalTranslation:
    def __init__(self, value: Union[float, tuple]) -> None:
        if type(value) is not tuple:
            value = (-value, value)
        self.value = value

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        W = indices.shape[4]
        for i in range(indices.shape[1]):
            value = np.random.uniform(
                low=self.value[0]*W, high=self.value[1]*W)
            indices[3, i] = indices[3, i] + value
        return indices


class RandomVerticalTranslation:
    def __init__(self, value: Union[float, tuple]) -> None:
        if type(value) is not tuple:
            value = (-value, value)
        self.value = value

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        H = indices.shape[3]
        for i in range(indices.shape[1]):
            value = np.random.uniform(
                low=self.value[0]*H, high=self.value[1]*H)
            indices[2, i] = indices[2, i] + value
        return indices


class RandomHorizontalScale:
    def __init__(self, scale: tuple) -> None:
        self.scale = scale

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        c = (np.array(indices.shape[3:], dtype='float32') - 1)/2
        for i in range(indices.shape[1]):
            scale = np.random.uniform(low=self.scale[0], high=self.scale[1])
            indices[3, i] = (indices[3, i] - c[1])/scale + c[1]
        return indices


class RandomVerticalScale:
    def __init__(self, scale: tuple) -> None:
        self.scale = scale

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        c = (np.array(indices.shape[3:], dtype='float32') - 1)/2
        for i in range(indices.shape[1]):
            scale = np.random.uniform(low=self.scale[0], high=self.scale[1])
            indices[2, i] = (indices[2, i] - c[0])/scale + c[0]
        return indices


class RandomRotation2D:
    def __init__(self, angle: Union[float, tuple]) -> None:
        if type(angle) is not tuple:
            angle = (-angle, angle)
        self.angle = np.radians(angle)

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        c = (np.array(indices.shape[3:], dtype='float32') - 1)/2
        for i in range(indices.shape[1]):
            angle = np.random.uniform(low=self.angle[0], high=self.angle[1])
            indices[2, i], indices[3, i] = (indices[2, i]-c[0])*np.cos(angle) - (indices[3, i]-c[1])*np.sin(
                angle) + c[0], (indices[2, i]-c[0])*np.sin(angle) + (indices[3, i]-c[1])*np.cos(angle) + c[1]
        return indices


class ElasticTransform:
    def __init__(self, alpha=600, sigma=30) -> None:
        self.alpha = alpha
        self.sigma = sigma

    def __call__(self, indices: np.ndarray) -> np.ndarray:
        for i in range(indices.shape[1]):
            # dx, dy = gaussian_filter(np.random.rand(*((2,) + indices.shape[3:]))*2 - 1, self.sigma, mode='constant', cval=0) * self.alpha
            dx = gaussian_filter(np.random.rand(
                *(indices.shape[3:]))*2 - 1, self.sigma, mode='constant', cval=0) * self.alpha
            dy = gaussian_filter(np.random.rand(
                *(indices.shape[3:]))*2 - 1, self.sigma, mode='constant', cval=0) * self.alpha
            indices[2, i], indices[3, i] = indices[2, i] + \
                dx, indices[3, i] + dy
        return indices


class Compose:
    def __init__(self, transformations: list, repeat: int = 1, shuffle: bool = False) -> None:
        self.transformations = transformations
        self.repeat = repeat
        if shuffle:
            random.shuffle(self.transformations)

    def __call__(self, input: np.ndarray, order: int = 0, mode: str = 'nearest', cval: float = 0.0, prefilter: bool = True) -> np.ndarray:
        augmentedData = np.empty((self.repeat*input.shape[0],) + input.shape[1:], dtype=input.dtype)
        for i in range(0, self.repeat*input.shape[0], input.shape[0]):
            ind = np.indices(input.shape)
            for T in self.transformations:
                ind = T(ind)
            augmentedData[i:i+input.shape[0]] = map_coordinates(input, ind, order=order, mode=mode, cval=cval, prefilter=prefilter)
        return augmentedData


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    from utils import loadNifti

    test = loadNifti([os.path.join('./MICCAI_BraTS2020_TrainingData/BraTS20_Training_001', name)
                      for name in os.listdir('./MICCAI_BraTS2020_TrainingData/BraTS20_Training_001')])
    test = np.rollaxis(test, -1,)[70:80]

    T = Compose([
        RandomVerticalFlip(0.5),
        RandomVerticalTranslation(0.1),
        RandomHorizontalTranslation(0.1),
        RandomVerticalScale((0.9, 1.1)),
        RandomHorizontalScale((0.9, 1.1)),
        RandomVerticalShear(0.05),
        RadnomHorizontalShear(0.05),
        RandomRotation2D(15),
        ElasticTransform(alpha=600, sigma=30),
    ], repeat=5, shuffle=False)
    
    ot = T(test)
    print(ot.shape)
    plt.figure()
    plt.imshow(ot[17, 0])
    plt.figure()
    plt.imshow(ot[18, 0])
    plt.figure()
    plt.imshow(ot[19, 0])
    plt.show()
