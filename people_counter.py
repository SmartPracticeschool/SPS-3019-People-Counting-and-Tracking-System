from packages.centroidtracker import CentroidTracker
from packages.trackableobject import TrackableObject
from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(
    r"modelfiles/MobileNetSSD_deploy.prototxt", r"modelfiles/MobileNetSSD_deploy.caffemodel")

if not (r"videos/example_01.mp4", False):
    print("[INFO] starting video stream...")
    vs = VideoStream(src=0).start()
    time.sleep(2.0)

else:
    print("[INFO] opening video file...")
    vs = cv2.VideoCapture("videos/example_01.mp4")

writer = None
W = None
H = None
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
trackers = []
trackableObjects = {}
totalFrames = 0
totalDown = 0
totalUp = 0
skip_frames = 30
fps = FPS().start()
