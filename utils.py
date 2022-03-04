import os
import cv2
from PIL import Image, ImageOps
from PIL import GifImagePlugin
import torch
import numpy as np
from torchvision import transforms

transform = transforms.Compose([
    transforms.ToTensor()
])


def keep_image_size_open(path, size=(256, 256)):
    img = Image.open(path)
    longestSide = max(img.size)
    _img = Image.new('RGB', (longestSide, longestSide), (0, 0, 0))
    _img.paste(img, (0, 0))
    _img = _img.resize(size)
    return _img


def keep_image_size_open_gray(path, size=(256, 256)):
    img = Image.open(path)
    img = ImageOps.grayscale(img)
    longestSide = max(img.size)
    mask = Image.new('P', (longestSide, longestSide))
    mask.paste(img, (0, 0))
    mask = mask.resize(size)
    return mask


def gray2RGB(img):
    out_img = torch.cat((img, img, img), 0)
    return out_img

def gray2Binary(img):
    img = np.array(img)
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            if img[i, j] > 0:
                img[i, j] = 255
    return Image.fromarray(img)


def if_point_in_corner(masks, i, j):
    if i ==0 or i == masks.shape[0] - 1 or j == 0 or j == masks.shape[1] - 1:
        return True
    else:
        return False

def extract_boundary(masks):
    offset = np.array([[0, 1], [1, 0], [-1, 0], [0, -1]])
    boundary = np.zeros(masks.shape, dtype=np.uint8)
    for i in range(masks.shape[0]):
        for j in range(masks.shape[1]):
            if masks[i, j] == 255:
                if not if_point_in_corner(masks, i, j):
                    flag = 0
                    for k in range(offset.shape[0]):
                        if masks[i + offset[k, 0], j+ offset[k, 1]] != 255:
                            flag = 1
                            break
                    if flag:
                        boundary[i, j] = 255       
                else:
                    boundary[i, j] = 255    
    return boundary

def annotation_img(img, boundary):
    for i in range(boundary.shape[0]):
        for j in range(boundary.shape[1]):
            if boundary[i, j] == 255:
                img[i, j] = [0, 0, 255]
    return img

                

if __name__ == '__main__':
    image = keep_image_size_open_gray(r'C:\Users\RTX 3090\Desktop\WangYichong\Data\Masks\Multi-institutional Paired Expert Segmentations MNI images-atlas-annotations\3_Annotations_MNI\CWRU\W1\W1_1996.10.25_CWRU_labels_AX\108.png')
    gimage = ImageOps.grayscale(image)
    ggimage = np.array(gimage)
    io.imshow(ggimage)
    plt.show()
    print(gimage.size)
    ggimage = gray2Binary(gimage)
    ggimage = np.array(ggimage)
    io.imshow(ggimage)
    plt.show()
