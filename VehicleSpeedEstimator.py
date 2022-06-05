# This is a simple Vehicle Speed Estimator created by Rodwin Alvarez
# Submitted on June 6, 2022 for Unleashed Live

from Detector.detector import Detector
from Tracker.sort import Sort
from SpeedEstimator.estimator import Estimator
import cv2, time, argparse
import numpy as np


def main(args):
    cap = cv2.VideoCapture(args.input)
    attrs = {} # Dictionary of vehicle Attributes
    ret, frame = cap.read() # VideoProcessor
    detector = Detector(frame) # Vehicle detector initialize
    speedometer = Estimator(frame,detector.width, detector.height) # Speed Estimator initialize
    mot_tracker = Sort(max_age=1,
                       min_hits=3,
                       iou_threshold=.3) # tracker initialize

    frame_no = 1
    frame_counter=0
    start = time.time()
    fps = 0
    while ret:
        img = frame
        detections, bboxes = detector.detect_image(img)
        trackers = mot_tracker.update(np.array(bboxes))
        speedometer.register_attributes(trackers, attrs)
        speedometer.project_points(attrs)
        image = detector.draw_bbox(trackers,img,attrs)
        if time.time() - start >= 1:
            start = time.time()
            fps = frame_counter
            frame_counter = 0
        cv2.putText(image, f'FPS: {fps}', (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    [255, 255, 0], 1)
        cv2.imshow('demo', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        ret, frame = cap.read()
        frame_no+=1
        frame_counter +=1
    cap.release()
    cv2.destroyWindow('demo')
    print('Video Ended!')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Parser for Speed Estimator')
    ap.add_argument("--input", "-i",default='./video_01.mp4')
    main(ap.parse_args()) # Main driver program
