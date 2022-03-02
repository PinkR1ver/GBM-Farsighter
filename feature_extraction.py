import os
from sys import exc_info
import pandas as pd
from six import u
from radiomics import featureextractor
import SimpleITK as sitk

basePath = os.path.dirname(__file__)

def ROI_not_one_dim(image_array):
    flag_i = False
    flag_j = False
    for val in (85, 170):
        image_array[image_array == val] = 255
    for i in range(image_array.shape[0] - 1):
        for j in range(image_array.shape[1] - 1):
            if(image_array[i + 1, j] == 255 and image_array[i, j] == 255):
                flag_i = True
                if(flag_i and flag_j):
                    return True
            if(image_array[i, j] == 255 and image_array[i, j + 1] == 255):
                flag_j = True
                if(flag_i and flag_j):
                    return True
    return False

def extract_feature(imgPath, maskPath):
    params = os.path.join(basePath, 'Params.yaml')
    extractor = featureextractor.RadiomicsFeatureExtractor(params)
    feature = extractor.execute(imgPath, maskPath)
    return feature

if __name__ == '__main__':
    pass