import os
import cv2
import random
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


def loadAssets():
    _metadataPath = r"data/metadata.csv"
    _metadataRGBPath = r"data/metadata_rgb_only.csv"
    _healthyPath = r"data/Brain Tumor Data Set/Brain Tumor Data Set/Healthy/"
    _tumorPath = r"data/Brain Tumor Data Set/Brain Tumor Data Set/Brain Tumor/"

    _df_metadata = pd.read_csv(_metadataPath)
    _df_metadataRGB = pd.read_csv(_metadataRGBPath)

    assets = [
        _healthyPath,
        _tumorPath,
        _df_metadata,
        _df_metadataRGB,
    ]

    return assets


def loadImages(path: str):
    _images = []
    for img in os.listdir(path):
        assert img is not None, f"Could not load img ({img})."
        _images.append(
            cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE)
        )
    return _images


def displayImage(img, stack=1):
    '''
        img: is the image list (List<Mat>) that will serve as a source for displaying

        stack: is the number of images that should be displayed in one kernel (Maximum is 5)

    '''
    assert img is not None, f"No image. Try again."

    assert stack >= 1 and stack <= 5, f"You can only display 1 up to 5 images at a time. Stack should not be attributed a negative number or zero."

    randomIndexes = random.sample(range(len(img)-1), stack)
    subset = [img[i] for i in randomIndexes]
    stacked_subset = np.hstack(subset)

    cv2.imshow(f'Images', stacked_subset)

    cv2.waitKey()

    return


def preProcess(imageList: list, eqHistogram=False, resize=()):

    if (eqHistogram):
        # may apply to the list directly
        for img in imageList:
            img = cv2.equalizeHist(img)

    if (resize):
        # needs to resize one by one
        imageList = [cv2.resize(_, resize) for _ in imageList]

    return imageList


def splitTrainTestValidation() -> None:
    assets = loadAssets()

    healthyFileNames = os.listdir(assets[0])
    tumorFileNames = os.listdir(assets[1])

    allFiles = healthyFileNames + tumorFileNames
    labels = [0] * len(healthyFileNames) + [1] * len(tumorFileNames)

    trainFiles, testFiles, trainLabels, testLabels = train_test_split(
        allFiles,
        labels,
        test_size=0.2,
        random_state=42,
    )

    trainFiles, valFiles, trainLabels, valLabels = train_test_split(
        trainFiles,
        trainLabels,
        test_size=0.15,
        random_state=42,
    )

    return print(
        "\n\n",
        len(allFiles),
        "\t",
        len(allFiles)/len(allFiles),
        "\n\n",
        len(trainFiles),
        "\t",
        len(trainFiles)/len(allFiles),
        "\n\n",
        len(testFiles),
        "\t",
        len(testFiles)/len(allFiles),
        "\n\n",
        len(valFiles),
        "\t",
        len(valFiles)/len(allFiles),
        "\n\n",
    )


if (__name__ == "__main__"):
    splitTrainTestValidation()
