import matplotlib.path as mplPath
from Utilities.VehicleAttributes import Attributes
import numpy as np
import math, cv2


class Estimator:
    '''
    This class handles the speed estimation of any object detected from the detector module. It changes the video
    perspective to a birds eye view to get a better estimate reading on the object movements based on an anchor point
    (e.g. centroid). The speed is then estimated by taking the distance travelled in the transformed space over the
    number of frames elapsed and then converted into kilometers per hour.

    To handle the required speed attributes of each tracked object, this class initiates the Attributes class to store
    the unique speed attributes of each tracked vehicle which is used for the speed estimation.
    '''
    def __init__(self,frame,width,height):
        self.points = np.float32(((241, 220), (505, 231), (904, 587), (249, 538)))
        self.roi = mplPath.Path(self.points)
        self.dst = np.float32(((0, 0), (88, 0), (88, 414), (0, 414)))
        self.M = cv2.getPerspectiveTransform(self.points, self.dst)
        self.xs = frame.shape[1] / width
        self.ys = frame.shape[0] / height
        self.max_count = 15

    def register_attributes(self,dets,attrs):
        '''
        Register speed attributes for each tracked object. All updates are done directly on the attrs object.
        :param dets: tracked objects provided by tracking module
        :param attrs: dictionary of VehicleAttributes
        :return: None
        '''
        for d in dets:
            id = d[4]
            x1, y1, x2, y2 = d[0] * self.xs, d[1] * self.ys, d[2] * self.xs, d[3] * self.ys
            xmid = int(x1)
            ymid = int(y1)
            cur_center = [xmid, ymid]
            if self.roi.contains_point(cur_center):
                if attrs.get(id):
                    if cur_center[1] > attrs[id].cur_center[1]:
                        attrs[id].cur_center = cur_center
                    attrs[id].counter += 1
                else:
                    cur_center = [xmid, ymid]
                    attrs[id] = Attributes()
                    attrs[id].cur_center = cur_center
                    attrs[id].prev_center = cur_center
                    attrs[id].counter += 1
            elif attrs.get(id):
                attrs[id].speed = []
                attrs[id].avg_speed = 0.0
                attrs[id].cur_center = cur_center

    def project_points(self,attrs):
        '''
        Project the points of tracked vehicles into the transformed perspective and perform speed estimation on the
        tracked objects. All updates are done directly on the attrs object.
        :param attrs: dictionary of VehicleAttributes
        :return: None
        '''
        for k, v in attrs.items():
            if v.counter >= self.max_count:
                centers = np.float32(np.array([[v.prev_center], [v.cur_center]]))
                transform_center = np.int32(cv2.perspectiveTransform(centers, self.M)).tolist()
                speed = v.speed
                xdiff = (transform_center[1][0][0] - transform_center[0][0][0]) ** 2
                ydiff = (transform_center[1][0][1] - transform_center[0][0][1]) ** 2
                cur_speed = math.sqrt(xdiff + ydiff) / (self.max_count * 2 / 30)
                if cur_speed < 10:
                    cur_speed = 0
                if len(speed) < 2:
                    speed.append(cur_speed)
                else:
                    speed.pop(0)
                    speed.append(cur_speed)
                v.speed = speed
                v.avg_speed = round((sum(speed) / len(speed)) * 1.098, 2)
                v.counter = 0
                v.prev_center = v.cur_center
