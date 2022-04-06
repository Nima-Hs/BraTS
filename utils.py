import os
import shutil
import random
from concurrent.futures import ThreadPoolExecutor
from typing import Union

import nibabel as nib
import numpy as np
from tqdm import tqdm


def loadNifti(path: Union[str, list, tuple], precision=np.single) -> np.ndarray:
    if type(path) == str:
        return precision(nib.load(path).get_fdata())
    else:
        data0 = loadNifti(path[0], precision)
        shape = (len(path),) + data0.shape
        data = np.empty(shape=shape, dtype=precision)
        data[0] = data0
        with ThreadPoolExecutor() as Executor:
            res = Executor.map(loadNifti, path[1:], len(path[1:])*[precision])
        for i, data0 in enumerate(res):
            data[i+1] = data0
        return data

def getAffine(path: Union[str, list, tuple]) -> np.ndarray:
    if type(path) == str:
        return nib.load(path).affine
    else:
        affines = []
        for p in path:
            affines.append(nib.load(p).affine)
        if not all((affine == affines[0]).all() for affine in affines):
            raise 'affines are not equal'
        return affines[0]

def getPathsFromRoot(bratsTrainingDirectory: str, keys: tuple = None):
    paths = []
    for root, dirs, files in os.walk(bratsTrainingDirectory, True):
        if files and not dirs:
            paths.append([os.path.join(root, f) for f in files if f.endswith('.nii.gz')])

    if keys is not None:
        l = []
        for i, key in enumerate(keys):
            for j, p in enumerate(paths[0]):
                if p.endswith(key + '.nii.gz'):
                    if i != j:
                        l.append([i, j])
                    continue
        for i, p in enumerate(paths):
            q = p.copy()
            for j in l:
                q[j[0]] = p[j[1]]
            paths[i] = q
    return paths

def separateData(bratsTrainingDirectory: str, validationFraction: float, testFraction: float, keys: tuple = None, shuffle: bool = False):
    paths = getPathsFromRoot(bratsTrainingDirectory, keys)
    if shuffle:
        random.shuffle(paths)
    valSize = round(validationFraction*len(paths))
    testSize = round(testFraction*len(paths))
    trainingPaths = paths[testSize + valSize:]
    validationPaths = paths[testSize:testSize + valSize]
    testPaths = paths[:testSize]
    return trainingPaths, validationPaths, testPaths


def splitLabels(seg: np.ndarray) -> np.ndarray:
    complete = seg != 0
    core = np.logical_or(seg == 1, seg == 4)
    enhancing = seg == 4
    return np.stack((complete, core, enhancing))

def mergeLabels(WT: np.ndarray, TC: np.ndarray, ET: np.ndarray) -> np.ndarray:
    return WT*(1-TC)*(1-ET)*2 + WT*TC*(1-ET) + WT*TC*ET*4

# def 

def zScore(input: np.ndarray) -> np.ndarray:
    input = (input - np.mean(input, axis=(1, 2, 3), keepdims=True)) / \
        np.std(input, axis=(1, 2, 3), keepdims=True)
    return input

def removeEmptySlices(input: np.ndarray) -> np.ndarray:
    return input[:, :, :, np.any(input, axis=(0, 1, 2))]


def saveExample(example: np.ndarray, name: str, precision=np.single) -> None:
    np.savez_compressed(name, input=precision(example[:4]), mask=example[4:].astype(bool))

def createDataset(paths: Union[list, tuple], directory: str, precision=np.single, batchSize: int = 4, deleteEmptySlices: bool = False, augment: bool = False, transforms=None) -> None:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)
    index = 0
    zfill = int(np.ceil(np.log10(155*len(paths))) if not augment else np.ceil(np.log10(155*len(paths)*transforms.repeat)))
    f1 = lambda item: np.rollaxis(np.concatenate([zScore(item[:-1]), splitLabels(item[-1])]), -1)
    for i in tqdm(range(0, len(paths), batchSize)):
        with ThreadPoolExecutor() as Executor:
            res = Executor.map(loadNifti, paths[i:i+batchSize])
            if deleteEmptySlices:
                res = Executor.map(removeEmptySlices, res)
            res = Executor.map(f1, res)
            # res = np.concatenate([np.rollaxis(np.concatenate([zScore(item[:-1]), splitLabels(item[-1])]), -1) for item in res])
            if augment:
                res = Executor.map(transforms, res)
            for item in res:
                # Executor.map(lambda it, i: saveExample(it, directory, i, zfill), item, range(index, index+item.shape[0]))
                Executor.map(saveExample, item, [os.path.join(directory, str(j).zfill(zfill)) for j in range(index, index+item.shape[0])], item.shape[0]*[precision])
                index += item.shape[0]

def saveNifti(data: np.ndarray, affine: np.ndarray, path: str) -> None:
    nib.save(nib.Nifti1Image(data, affine=affine), path)

if __name__ == '__main__':
    import augmentation as aug

    # transforms = aug.Compose([
    #     aug.RandomVerticalFlip(0.5),
    #     aug.RandomVerticalTranslation(0.1),
    #     aug.RandomHorizontalTranslation(0.1),
    #     aug.RandomVerticalScale((0.9, 1.1)),
    #     aug.RandomHorizontalScale((0.9, 1.1)),
    #     aug.RandomVerticalShear(0.05),
    #     aug.RadnomHorizontalShear(0.05),
    #     aug.RandomRotation2D(15),
    #     aug.ElasticTransform(alpha=600, sigma=30),
    # ], repeat=10)
    # # seed = int(time())//10
    # seed = 161426206
    # print(f'seed = {seed}')
    # random.seed(seed)
    # # trainingPaths, validationPaths, testPaths = separateData(
    # #     './MICCAI_BraTS2020_TrainingData', 0.1, 0.1, keys=['flair', 't1', 't1ce', 't2', 'seg'])
    # # createDataset(testPaths, './validation16', precision=np.half, batchSize=1, deleteEmptySlices=True, augment=True, transforms=transforms)
    # a = np.load('./validation16/00000.npz')
    # print(a['input'].dtype)
