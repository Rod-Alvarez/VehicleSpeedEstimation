from Detector.darknet import *
import cv2

class Detector:
    '''
    This class handles a quick and easy way to access the yolo detector implemented in its native darknet framework.
    It uses the provided darknet.py file as the main interface file to interact with the darknet and yolo_cpp_dll
    libraries compiled in darknet. I have also created the filter_detections and draw_bbox modules to further suppport
    the requirement for the speed estimator.
    '''
    def __init__(self,img):
        self.config = './Detector/yolov3.cfg'
        self.weights = './Detector/yolov3.weights'
        self.data = './Detector/coco.data'
        self.thresh = 0.25
        self.network, self.class_names, self.colors = load_network(self.config,self.data,self.weights,batch_size=1)
        self.width = network_width(self.network)
        self.height = network_height(self.network)
        self.xs = img.shape[1] / self.width
        self.ys = img.shape[0] / self.height

    @staticmethod
    def filter_detections(dets):
        '''
        Filter out irrelevant classes and detection errors with wide bbox sizes.
        :param dets: raw detection output from detect_image
        :return: filtered detections, formatted bboxes fit for tracker
        '''
        temp = []
        bboxes = []
        for d in dets:
            if d[0] == 'car':
                box = list(d[2])
                if box[2] < 30:
                    box[2] += box[0]
                    box[3] += box[1]
                    box.append(float(d[1]))
                    bboxes.append(box)
                    temp.append(d)
        return temp, bboxes

    def detect_image(self,img):
        '''
        Image detection pipeline using darknet yolo
        :param img: image which detections are to be done
        :return: list of filtered detections from darknet, list of formatted bboxes fit for tracker
        '''
        darknet_image = make_image(self.width, self.height, 3)
        image_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        copy_image_from_bytes(darknet_image, image_resized.tobytes())
        detections = detect_image(self.network, self.class_names, darknet_image, thresh=self.thresh)
        free_image(darknet_image)
        return self.filter_detections(detections)

    def draw_bbox(self,detections, img, attrs):
        '''
        Visualization of detections with bboxes, tracking ID, and centroid. Also visualizes the region of interest
        specified for the video scene
        :param detections: tracked objects returned by the tracking module
        :param img: source image where visualizations are to be drawn
        :param attrs: dictionary of VehicleAttributes
        :return: updated image with all the visualizations drawn
        '''
        for detection in detections:
            id = detection[4]
            xmin, ymin, xmax, ymax = detection[0], \
                                     detection[1], \
                                     detection[2], \
                                     detection[3]
            w = (xmax - xmin) / 2
            h = (ymax - ymin) / 2
            if w > 30:
                continue
            pt1 = (int((xmin - w) * self.xs), int((ymin - h) * self.ys))
            pt2 = (int((xmax - w) * self.xs), int((ymax - h) * self.ys))
            cv2.rectangle(img, pt1, pt2, (0, 255, 0), 2)
            cv2.putText(img,
                        'car' + " [" + str(detection[4]) + "]",
                        (pt1[0], pt1[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, .5,  # 0.5
                        [0, 255, 0], 2)
            img = cv2.line(img, (241, 220), (249, 538), (0, 0, 255), 2)
            img = cv2.line(img, (241, 220), (505, 231), (0, 0, 255), 2)
            img = cv2.line(img, (904, 587), (249, 538), (0, 0, 255), 2)
            img = cv2.line(img, (904, 587), (505, 231), (0, 0, 255), 2)
            if attrs.get(id):
                cv2.putText(img, str(attrs[id].avg_speed), (int(xmin * self.xs) - 20, int(ymin * self.ys) - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, .5,  # 0.5
                            [0, 255, 255], 2)
                img = cv2.circle(img, attrs[id].cur_center, 3, (255, 0, 0), -1)
        return img
