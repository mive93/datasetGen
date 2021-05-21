import cv2
import os
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Check YOLO labels")
parser.add_argument("--input_stream", type=str, default="data/yolo_test.mp4",
                    help="RTSP stream or video path from wich to extract images")
parser.add_argument("--image_dir", type=str, default="images",
                    help="folder where to store the images")
parser.add_argument("--s", type=int, default=60,
                    help="interval of seconds to store images")
args = parser.parse_args()


if not os.path.exists(args.image_dir):
    os.makedirs(args.image_dir)

cap = cv2.VideoCapture(args.input_stream)
if (cap.isOpened()== False): 
  print("Error opening video stream or file")

fps = cap.get(cv2.CAP_PROP_FPS)
print(fps)

counter = 0
index = 0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == True:
        cv2.imshow('Frame',frame)
        counter += 1
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
        if counter == args.s * fps:
            cv2.imwrite(os.path.join(args.image_dir, str(index)+".png"), frame)
            index += 1
            counter = 0
    else: 
        break

cap.release()
cv2.destroyAllWindows()
