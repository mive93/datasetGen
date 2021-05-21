import json
import sys
import os
import argparse
import sys
import cv2
import math
import numpy as np

parser = argparse.ArgumentParser(description="Check YOLO labels")
parser.add_argument("--images_file", type=str, default="dataset.txt",
                    help="path to a txt file with all the paths of the images to check")
parser.add_argument("--labels_dir", type=str, default="labels",
                    help="folder where to find the labels")
args = parser.parse_args()

cur_file = open(args.images_file, 'r')
lines = cur_file.readlines()

for line in lines:
    image_path = line.split("\n")[0]
    label_name = image_path.split('/')[-1].replace("png", "txt", 1)
    label_path = os.path.join(args.labels_dir, label_name)

    # print(image_path, label_path)

    img = cv2.imread(image_path, 1)
    height, width, chan = img.shape

    # print(width, height)
    
    cur_labels = open(label_path, 'r')
    llines = cur_labels.readlines()
    for a in llines:
        values = a.split("\n")[0].split(" ")
        # print(values)
        x = int(float(values[1]) * width)
        y = int(float(values[2]) * height)
        w = int(float(values[3]) * width)
        h = int(float(values[4]) * height)
        # print(x1,x2,y1,y2)
        img = cv2.rectangle(img, (x - w/2, y - h/2), (x + w/2 , y + h/2), (0, 0, 255), 2)

    # cv2.imwrite("test.png", img)
    # input()

    cv2.imshow('image', img)
    cv2.waitKey(0)
