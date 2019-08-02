# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
import sys
import numpy as np
import os.path

# Initialize the parameters
confThreshold = 0.7 # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold置信度阈值
inpWidth = 160  # Width of network's input image，改为320*320更快
inpHeight = 160  # Height of network's input image，改为608*608更准

# parser = argparse.ArgumentParser(description='Object Detection using YOLO in OPENCV')
# parser.add_argument('--image', help='Path to image file.')
# parser.add_argument('--video', help='Path to video file.')
# args = parser.parse_args()
type =0 #0表示读取视频 ，1表示读取图片
video_path = "../test/live/11.mp4"
image_path = '../test/1.jpg'
# Load names of classes
classesFile = "./model/voc_classes.txt"
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "./model/yolov3_tiny.cfg"
# modelWeights = "./model/trained_weights_stage_1.weights"
modelWeights = "./model/yolov3_tiny.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)  # 可切换到GPU,cv.dnn.DNN_TARGET_OPENCL，


# 只支持Intel的GPU,没有则自动切换到cpu

# Get the names of the output layers
def getOutputsNames(net):
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]


# Draw the predicted bounding box
def drawPred(classId, conf, left, top, right, bottom):
    # Draw a bounding box.
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)

    label = '%.2f' % conf

    # Get the label for the class name and its confidence
    if classes:
        assert (classId < len(classes))
        label = '%s:%s' % (classes[classId], label)

    # Display the label at the top of the bounding box
    labelSize, baseLine = cv.getTextSize(label, cv.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    top = max(top, labelSize[1])
    cv.rectangle(frame, (left, top - round(1.5 * labelSize[1])), (left + round(1.5 * labelSize[0]), top + baseLine),
                 (255, 255, 255), cv.FILLED)
    cv.putText(frame, label, (left, top), cv.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 1)


# Remove the bounding boxes with low confidence using non-maxima suppression
def postprocess(frame, outs):
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    classIds = []
    confidences = []
    boxes = []
    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left, top, left + width, top + height)


# Process inputs
# winName = 'Deep learning object detection in OpenCV'
# cv.namedWindow(winName, cv.WINDOW_NORMAL)


if (type):
    # Open the image file
    if not os.path.isfile(image_path):
        print("Input image file ", image_path, " doesn't exist")
        sys.exit(1)
    cap = cv.imread(image_path)
    outputFile = './result/1.jpg'
else:
    # Open the video file
    if not os.path.isfile(video_path):
        print("Input video file ", video_path, " doesn't exist")
        sys.exit(1)
    cap = cv.VideoCapture(video_path)
    outputFile = './result/result.avi'

# Get the video writer initialized to save the output video
if (not type):
    vid_writer = cv.VideoWriter(outputFile, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30,
                                (round(cap.get(cv.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv.CAP_PROP_FRAME_HEIGHT))))

while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    if hasFrame:
        img_width = frame.shape[1]
        img_height = frame.shape[0]
        if img_width > img_height:
            padding_len = int((img_width - img_height) / 2)
            frame = cv.copyMakeBorder(frame, padding_len, padding_len, 0, 0, cv.BORDER_CONSTANT)
        else:
            padding_len = int((img_height - img_width) / 2)
            frame = cv.copyMakeBorder(frame, 0, 0, padding_len, padding_len, cv.BORDER_CONSTANT)

    # Stop the program if reached end of video
    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        break


    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(frame, 1 / 255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    # Remove the bounding boxes with low confidence
    postprocess(frame, outs)

    # Put efficiency information. The function getPerfProfile returns the overall time for inference(t) and the timings for each of the layers(in layersTimes)
    t, _ = net.getPerfProfile()
    label = 'Inference time: %.2f ms' % (t * 1000.0 / cv.getTickFrequency())
    cv.putText(frame, label, (0, 15), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255))
    cv.imshow('tets', frame)
    cv.waitKey(20)
    # Write the frame with the detection boxes
    # if (not type):
    #     cv.imwrite(outputFile, frame.astype(np.uint8))
    # else:
    #     vid_writer.write(frame.astype(np.uint8))

    # cv.imshow(winName, frame)