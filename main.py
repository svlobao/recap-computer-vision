import os
import cv2
import pandas as pd


def loadAssets():
    _metadataPath = r"data/metadata.csv"
    _metadataRGBPath = r"data/metadata_rgb_only.csv"
    _healthyPath = r"data/Brain Tumor Data Set/Brain Tumor Data Set/Healthy"
    _tumorPath = r"data/Brain Tumor Data Set/Brain Tumor Data Set/Brain Tumor"

    assets = [
        _metadataPath,
        _metadataRGBPath,
        _healthyPath,
        _tumorPath
    ]

    return assets


def loadImages(path: str):
    _images = []
    for img in os.listdir(path):
        assert img is not None, f"Could not load img ({img})."
        _images.append(
            cv2.imread(os.path.join(path, img))
        )
    return _images


def displayImage(img, startPos=-2, ):
    assert img is not None, f"No image. Try again."
    cv2.imshow('Sample', img)
    cv2.waitKey()
    return


def getHistogram(img):
    _hist = cv2.calcHist(img, [0], None, [256], [0, 256])
    return _hist


if (__name__ == "__main__"):

    assets = loadAssets()

    df_metadata = pd.read_csv(assets[0])
    df_metadataRGB = pd.read_csv(assets[1])

    healthyImages = loadImages(assets[2])
    tumorImages = loadImages(assets[3])

    histHealthyImages = getHistogram(healthyImages)
    histTumorImages = getHistogram(tumorImages)

    print(
        len(histHealthyImages),
        len(histTumorImages),
        type(histHealthyImages),
        type(histTumorImages),
    )
