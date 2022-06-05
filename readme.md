# Unleashed Technical Test by Rodwin Alvarez
## Contents
- General Approach
- Pre-requisites and Installation
- Code Structure and Explanation
- Areas for Improvement

## General Approach
The video provided is of an HD (1280x720) video intersection with a field of view centering on the vehicles going southbound
(from the camera's perspective) however, the vehicles going eastbound, northbound, westward on the 
intersection as well as the side street turning into the northbound road are visible. The task is to 
detect and track the vehicles and determine their speed as they move around the camera frame. 

The criteria provided for this assessment is as follows:
- Code Quality
- Quality of the results: detection, tracking and speed accuracy
- Usability of output data
- Model performance: Output FPS
- Scalability of the solution

As a proof of concept, I have chosen to just focus on the southbound vehicles only when estimating the speed 
of these vehicles though I have still included the detection and tracking of all the vehicles that can be
seen within the frame. It is possible to have also included the northbound speed estimation for this video 
but requires additional time to configure and fine tune everything properly.

With these in mind, I then broke down the problem into 4 main bodies of work namely:
- Detection
- Tracking
- Perspective correction
- Speed Estimation

For the detection algorithm, I have chosen to go with yolov3. Though it is quite old, it is still very good
with basic detection tasks and processes with manageable speeds. I did not have to go to the latest versions
of detectors available as yolov3 (https://github.com/AlexeyAB/darknet) was already fit for this purpose. 
Even if I am only focusing on the southbound vehicles, I purposely did not mask the surrounding areas to do 
a good stress test on the model's performance as the volume of cars increase. This is to give me an idea of
worst case performance output.

As to the tracking module, I have chosen to go with SORT (https://github.com/abewley/sort) which is a tracking
algorithm that uses kalman filters to predict the tracked objects position in hopes to have a better IOU
match than older tracking methods. I went with this as the video shows cars that queue up behind the stop
line potentially stacking the bounding boxes from the detectors together due to the angle of the camera
view. SORT in this case would perform better when compared with centroid tracking and limit tracking ID 
exchanges between vehicles.

Considering that the road is warped within the camera field of view, I decided to perform some perspective
corrections using homography where I marked a specific warped square shape region of interest (ROI) from the 
camera view. I tried to map the ROI best to the full length of the vertical solid white lines measuring 207' 
(feet I presume) as well as the full width of the horizontal white stopping line measuring 44'. With this 
estimated ROI, I then calculated a transformation matrix that can transform the ROI perspective in the frame 
into a matrix of 88x214 in size (doubling 44x207 for easier viewing). With this transformation matrix, I can 
then map vehicle centroid positions from the camera view into the transformed perspective space where I can
measure the distance traveled in pixels. In this setup, the rough conversion becomes ~2pixels per 1 foot.

Lastly, the speed detection module handled the calculation of the distance traveled per frame into a speed
measurement. This was done by taking the distance traveled over 15 frames which roughly is about .5 seconds
given that the raw FPS of the video is ~29FPS (I have made an assumption that the raw video FPS is real 
world). With these 2 info, I am able to calculate the feet (pixes) / second speed metric of each vehicle. 
Converting this to km/h became just multiplying the value I get by 1.098 (1ft = .000305km, 3600s = 1h, total 
= 1.098)

Combining everything, I then created a quick opencv window to showcase the solution showing the vehicle 
detections (bounding boxes), tracking ID (upper left of bbox) and estimated speed in km/h (just above 
centroid). I have also posted a throughput reading in FPS in the bottom left corner of the window to monitor
output FPS. 

## Pre-requisites and Installation
For yoloV3, I decided to build it straight from within its home framework of darknet as it is reliably and 
consistently fast. To run darknet, you will need to set up and install the following:
- Python>=3.9
- NVIDIA GPU CC >=6.1 (Currently using GTX 1060, recommended CC >=7.5)
- CUDA w/ CuDNN
- opencv (build opencv_world with opencv_contrib extra modules)
- darknet

In installing darknet, it is also important to build the yolo_cpp_dll.dll as a dependency to be able to use 
the darknet.dll library. Once done, the following files are required to be in the same directory with the 
detector module that uses these libraries:
- darknet.dll
- opencv_world420.dll
- pthreadGC2.dll
- pthreadVC2.dll
- yolo_cpp.dll
- zlibwapi.dll (from Nvidia)

It is important to note that my GPU does not have any tensor cores which makes it unavailable to run yolo
in fp16. As such, I had to reduce the resolution of the input (256x256) to compensate and trade-off a bit of 
accuracy for speed to comply with the model performance criteria.

As to the weights, I have not done any additional training and I'm using the raw weights I downloaded 
directly from the yoloV3 darknet owner (https://github.com/AlexeyAB/darknet).

## Code Structure and Explanation
The main driver of the program is the main.py file in the main directory of the repo. This file handles the 
processing of the video input as well as timing the speed of the whole pipeline. This file is also the one
initializing the other modules that are required for this program to run specifically the detector, tracker, 
and the speed estimator classes. 

Under the main directory are 4 other subdirectories. Each houses other class definitions that are being used
by the program. This structure is to have easy tracing whenever bugs or issues arises as it makes it easier
to know the source class of the issue given its symptoms.

The Detector class handles all detector related tasks. It receives an image, makes detections from the image,
filters out unnecessary detections and can draw the detections back to the image. 

The tracker class is a direct use of the SORT tracker from https://github.com/abewley/sort. I did not make 
any edits from this file. 

The Speed estimator class handles both the perspective transform of the current position of the vehicles
and the speed calculation of each as they enter inside the ROI. This class requires a helper class
(Attributes class) that holds the position and speed related attributes of each vehicle being tracked. 

## Areas for Improvement
##### YoloV3
- The model can do with more training to reduce bbox jitters
- GPU CC >7 to be able to run on fp16 for faster processing at higher input resolutions

##### Tracking
- Make use of the predicted velocity attribute for better visualization (trajectory projection)
- incorporate extra attributes for better tracking

##### Speed Estimation
- Further investigation required on some weird values
- More fine-tuning required on calculation parameters
- Need to smoothen out positions over time to reduce jumping values
- Expand speed estimation on northbound lane
- Visualize mapped perspective

##### Others/General
- Implement code obfuscation for production setting and package as executable
- Improve management of configurations
- Add more configurable parameters for better over all flexibility

