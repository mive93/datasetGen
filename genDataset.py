import sys
sys.path.insert(1, 'darknet')

import argparse
import os
import glob
import random
import darknet
import time
import cv2
import numpy as np
import darknet
import json
import copy

def parser():
    parser = argparse.ArgumentParser(description="YOLO Object Detection Data Generation")
    parser.add_argument("--dataset_location", type=str, default="images",
                        help="image to annotate folder path")
    parser.add_argument("--outdir", type=str, default="labels",
                        help="folder where to write the labels")
    parser.add_argument("--weights", default="yolov4.weights",
                        help="yolo weights path")
    parser.add_argument("--config_file", default="darknet/cfg/yolov4.cfg",
                        help="path to config file")
    parser.add_argument("--data_file", default="darknet/cfg/coco.data",
                        help="path to data file (e.g. coco.data)")
    parser.add_argument("--thresh", type=float, default=.25,
                        help="remove detections with lower confidence")
    return parser.parse_args()


def check_arguments_errors(args):
    assert 0 < args.thresh < 1, "Threshold should be a float between zero and one (non-inclusive)"
    if not os.path.exists(args.config_file):
        raise(ValueError("Invalid config path {}".format(os.path.abspath(args.config_file))))
    if not os.path.exists(args.weights):
        raise(ValueError("Invalid weight path {}".format(os.path.abspath(args.weights))))
    if not os.path.exists(args.data_file):
        raise(ValueError("Invalid data file path {}".format(os.path.abspath(args.data_file))))
    if args.dataset_location and not os.path.exists(args.dataset_location):
        raise(ValueError("Invalid image path {}".format(os.path.abspath(args.dataset_location))))

def load_images(images_path):
    """
    If image path is given, return it directly
    For txt file, read it and return each line as image path
    In other case, it's a folder, return a list with names of each
    jpg, jpeg and png file
    """
    input_path_extension = images_path.split('.')[-1]
    if input_path_extension in ['jpg', 'jpeg', 'png']:
        return [images_path]
    elif input_path_extension == "txt":
        with open(images_path, "r") as f:
            return f.read().splitlines()
    else:
        return glob.glob(
            os.path.join(images_path, "*.jpg")) + \
            glob.glob(os.path.join(images_path, "*.png")) + \
            glob.glob(os.path.join(images_path, "*.jpeg"))




def image_detection(image_path, network, class_names, class_colors, thresh):
    # Darknet doesn't accept numpy images.
    # Create one with image we reuse for each detect
    width = darknet.network_width(network)
    height = darknet.network_height(network)
    darknet_image = darknet.make_image(width, height, 3)

    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (width, height),
                               interpolation=cv2.INTER_LINEAR)

    darknet.copy_image_from_bytes(darknet_image, image_resized.tobytes())
    detections = darknet.detect_image(network, class_names, darknet_image, thresh=thresh)
    new_dets = []
    for det in detections:
        new_x1 = det[2][0]*image.shape[1]/width
        new_y1 = det[2][1]*image.shape[0]/height
        new_x2 = det[2][2]*image.shape[1]/width
        new_y2 = det[2][3]*image.shape[0]/height
        new_det = (new_x1, new_y1, new_x2, new_y2)
        new_dets.append((det[0], det[1], new_det))
    darknet.free_image(darknet_image)


    image = darknet.draw_boxes(new_dets, image, class_colors)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB), new_dets

def get_iou(bb1, bb2):
    """
    Calculate the Intersection over Union (IoU) of two bounding boxes.

    Parameters
    ----------
    bb1 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x1, y1) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner
    bb2 : dict
        Keys: {'x1', 'x2', 'y1', 'y2'}
        The (x, y) position is at the top left corner,
        the (x2, y2) position is at the bottom right corner

    Returns
    -------
    float
        in [0, 1]
    """
    assert bb1['x1'] < bb1['x2']
    assert bb1['y1'] < bb1['y2']
    assert bb2['x1'] < bb2['x2']
    assert bb2['y1'] < bb2['y2']

    # determine the coordinates of the intersection rectangle
    x_left = max(bb1['x1'], bb2['x1'])
    y_top = max(bb1['y1'], bb2['y1'])
    x_right = min(bb1['x2'], bb2['x2'])
    y_bottom = min(bb1['y2'], bb2['y2'])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    # The intersection of two axis-aligned bounding boxes is always an
    # axis-aligned bounding box
    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    # compute the area of both AABBs
    bb1_area = (bb1['x2'] - bb1['x1']) * (bb1['y2'] - bb1['y1'])
    bb2_area = (bb2['x2'] - bb2['x1']) * (bb2['y2'] - bb2['y1'])

    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = intersection_area / float(bb1_area + bb2_area - intersection_area)
    assert iou >= 0.0
    assert iou <= 1.0
    return iou

def clickAndRemove(event, x, y, flags, param):
    global detections, img, class_colors, orig_det
    image = img.copy()
    #clicking on a detection with the left button means remove it because it's wrong
    if event == cv2.EVENT_LBUTTONDOWN:
        new_det = []
        for d in detections:
            if x >= d[2][0] - d[2][2]/2. and x <= d[2][0] + d[2][2]/2. and y >= d[2][1] - d[2][3]/2. and y <= d[2][1] + d[2][3]/2.:
                print (x,y)
            else:
                new_det.append(d)                

        detections = new_det
        image = darknet.draw_boxes(detections, image, class_colors)
        cv2.imshow('image', image)

    #a click with the right button will restore the original detections of the net (in case of user mistake)
    if event == cv2.EVENT_RBUTTONDOWN:
        detections = copy.deepcopy(orig_det)
        image = darknet.draw_boxes(detections, image, class_colors)
        cv2.imshow('image', image)

if __name__ == "__main__":

    args = parser()
    check_arguments_errors(args)

    #Yolo settings
    random.seed(3)  # deterministic bbox colors
    network, class_names, class_colors = darknet.load_network(
        args.config_file,
        args.data_file,
        args.weights
    )

    dataset_folder  = args.dataset_location

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    cv2.namedWindow("image")
    cv2.setMouseCallback("image", clickAndRemove)

    file_list = open("dataset.txt", "w")
    for filename in os.listdir(dataset_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            image_name = os.path.join(dataset_folder, filename)
            # print(image_name)

            img = cv2.imread(image_name)

            #yolo inference
            image, detections = image_detection(image_name, network, class_names, class_colors, args.thresh)
            
            #show image and let the user remove wrong labels
            cv2.imshow('image', image)
            orig_det = copy.deepcopy(detections)
            cv2.waitKey(0)

            #write correct labels to file
            img_name = filename.split('.')[0]
            if detections != []:
                f = open(args.outdir + "/" + img_name + ".txt", "w")
                for d in detections:
                    c = class_names.index(d[0])
                    x = float(d[2][0])/img.shape[1]
                    y = float(d[2][1])/img.shape[0]
                    w = float(d[2][2])/img.shape[1]
                    h = float(d[2][3])/img.shape[0]
                    if w < 0 or h < 0:
                        raise Exception('coord error')
                    f.write(str(c)+" "+str(x)+" "+str(y) +" "+str(w)+" "+str(h)+"\n")
                f.close()
                file_list.write(image_name+"\n")

    file_list.close()
            

            


