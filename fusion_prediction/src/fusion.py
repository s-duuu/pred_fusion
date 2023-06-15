#! /usr/bin/env python3

import rospy
import numpy as np
import cv2
import math

from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from fusion_prediction.msg import CameraBBox, CameraBBoxes

def quaternion2euluer(x, y, z, w):
    """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
    """
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return math.degrees(yaw_z) # in degrees

def local2global(local_pt, x_centroid, y_centroid, yaw):
    """
        Convert a local point into global point considering PointPillars output.
        1. x_centroid, y_centroid
        2. yaw (in degrees)
        Output is a !!!row vector!!! of a global point.
    """
    obj_local = np.array([local_pt[0], local_pt[1]]).T
    matrix = np.array([[-math.cos(math.radians(yaw)), -math.sin(math.radians(yaw))], [math.sin(math.radians(yaw)), -math.cos(math.radians(yaw))]])
    centroid = np.array([x_centroid, y_centroid]).T

    world_output = matrix @ obj_local + centroid

    return world_output

def calc_angle(vec1, vec2):
    """
        Calculate angle between two vectors using inner product.
    """
    inner_product = np.dot(vec1, vec2)
    
    size_1 = math.sqrt(np.dot(vec1, vec1))
    size_2 = math.sqrt(np.dot(vec2, vec2))

    return math.degrees(math.acos(inner_product/(size_1 * size_2)))

def iou(cam_bbox, lidar_bbox):
    """
        Calculate IOU between 2 bounding boxes
    """
    x1 = max(cam_bbox.xmin, lidar_bbox[0])
    y1 = max(cam_bbox.ymin, lidar_bbox[1])
    x2 = min(cam_bbox.xmax, lidar_bbox[2])
    y2 = min(cam_bbox.ymax, lidar_bbox[3])

    cam_bbox_area = (cam_bbox.xmax - cam_bbox.xmin) * (cam_bbox.ymax - cam_bbox.ymin)
    lidar_bbox_area = (lidar_bbox[2] - lidar_bbox[0]) * (lidar_bbox[3] - lidar_bbox[1])

    inter_area = np.maximum((x2 - x1), 0) * np.maximum((y2 - y1), 0)

    total_area = cam_bbox_area + lidar_bbox_area - inter_area

    iou = inter_area / total_area

    return iou

class fusion:
    def __init__(self):
        
        self.bridge = CvBridge()

        # 센서 퓨전 이후 최종 객체 정보 dict
        self.fusion_distance_dict = dict()
        self.fusion_velocity_dict = dict()
        self.fusion_heading_dict = dict()

        # 각 센서별 객체 검출 결과 list (bbox 정보)
        self.cam_bbox_list = []
        self.lidar_bbox_list = []

        # 센서 퓨전 이후 최종 객체별 bbox dict
        self.final_object_bbox_dict = dict()
        
        # dict 객체 id별 초기화
        for i in range(1, 11):
            self.fusion_distance_dict[i] = []
            self.fusion_velocity_dict[i] = []
            self.fusion_heading_dict[i] = []
            self.final_object_bbox_dict[i] = []
        
        # ROS node 초기화 & subscriber 선언
        rospy.init_node('fusion_node', anonymous=False)
        rospy.Subscriber('yolov5/detections', CameraBBoxes, self.camera_callback)
        rospy.Subscriber('yolov5/image_out', Image, self.image_callback)
        rospy.Subscriber('pillars/detections', BoundingBoxArray, self.lidar_callback)
        self.fusion_pub = rospy.Publisher('display/detections', BoundingBoxArray, queue_size=50)

    
    # YOLO 검출 결과 callback 함수
    def camera_callback(self, data):
        self.cam_bbox_list = data.bounding_boxes
        self.cam_bbox_num = len(self.cam_bbox_list)
         

    def image_callback(self, data):
        self.image = self.bridge.imgmsg_to_cv2(data, desired_encoding="bgr8")
        

    # PointPillars 검출 결과 callback 함수
    def lidar_callback(self, data):
        self.lidar_header_info = data.header
        self.lidar_bbox_list = data.boxes
        self.lidar_bbox_num = len(self.lidar_bbox_list)
        self.calib_sensor()
        
    
    # Image 평면에 투영할 LiDAR bbox 4개 point 선정 함수
    def bbox2globalBEV(self, lidar_bbox):
        """
            Convert Object local coordinate into global coordinate.
            
            Output is a matrix of four points with global 2D BEV coordinates.
        """
        x_centroid = lidar_bbox.pose.position.x
        y_centroid = lidar_bbox.pose.position.y
        z_centroid = lidar_bbox.pose.position.z

        yaw = quaternion2euluer(lidar_bbox.pose.orientation.x, lidar_bbox.pose.orientation.y, lidar_bbox.pose.orientation.z, lidar_bbox.pose.orientation.w)
        
        width = lidar_bbox.dimensions.x
        length = lidar_bbox.dimensions.y
        height = lidar_bbox.dimensions.z

        local_pt1 = (width/2, length/2)
        local_pt2 = (width/2, -length/2)
        local_pt3 = (-width/2, length/2)
        local_pt4 = (-width/2, -length/2)

        world_pts = []

        world_pts.append(local2global(local_pt1, x_centroid, y_centroid, yaw).tolist())
        world_pts.append(local2global(local_pt2, x_centroid, y_centroid, yaw).tolist())
        world_pts.append(local2global(local_pt3, x_centroid, y_centroid, yaw).tolist())
        world_pts.append(local2global(local_pt4, x_centroid, y_centroid, yaw).tolist())

        return world_pts

    
    def select_corners(self, lidar_bbox, world_pts):
        """
            Convert 2D BEV 4 points into 3D 4 points.
            After conversion, the minimum & maximum angle 4 points are selected.

            Output is a matrix of 4 points, and each point is a !!!row vector!!! of the matrix.
        """

        z_centroid = lidar_bbox.pose.position.z

        height = lidar_bbox.dimensions.z

        ref_vec = np.array([1, 0]).T
        angle_list = []

        angle_list.append(calc_angle(np.array(world_pts[0]), ref_vec))
        angle_list.append(calc_angle(np.array(world_pts[1]), ref_vec))
        angle_list.append(calc_angle(np.array(world_pts[2]), ref_vec))
        angle_list.append(calc_angle(np.array(world_pts[3]), ref_vec))

        min_idx = angle_list.index(min(angle_list))
        max_idx = angle_list.index(max(angle_list))

        corner_pts = np.array([[world_pts[min_idx][0], world_pts[min_idx][1], z_centroid - height/2], 
                               [world_pts[min_idx][0], world_pts[min_idx][1], z_centroid + height/2],
                               [world_pts[max_idx][0], world_pts[max_idx][1], z_centroid - height/2],
                               [world_pts[max_idx][0], world_pts[max_idx][1], z_centroid + height/2]
                               ])

        return corner_pts


    # 카메라 & 라이다 calibration
    def calib_sensor(self):
        """
            Function of calibration between Camera & LiDAR.
            1) Using intrinsic & extrinsic matrix, LiDAR bbox will be projected to the image plane.
            2) After projection, iou will be calculated. 
            If at least iou of one pair is over 0.65, the bbox will be remained. 
            Otherwise, the bbox will be removed.
        """
        intrinsic_matrix = np.array([[(self.image.shape[1]//2)/math.tan(0.5*math.radians(50)), 0, (self.image.shape[1]//2)], [0, (self.image.shape[1]//2)/math.tan(0.5*math.radians(50)), (self.image.shape[0]//2)], [0, 0, 1]])
        extrinsic_matrix = np.array([[-0.1736, -0.9848, 0, -1.08], [0, 0, -1, -0.5], [0.9848, -0.1736, 0, 0.368]])

        boxes = BoundingBoxArray()
        boxes.header = self.lidar_header_info

        cnt = 1

        self.remove_list = []

        after_fusion_box_list = []

        for lidar_bbox in self.lidar_bbox_list:

            yaw = quaternion2euluer(lidar_bbox.pose.orientation.x, lidar_bbox.pose.orientation.y, lidar_bbox.pose.orientation.z, lidar_bbox.pose.orientation.w)

            world_pts = self.bbox2globalBEV(lidar_bbox)

            corner_pts = self.select_corners(lidar_bbox, world_pts)

            proj_x_pts = []
            proj_y_pts = []

            for corner in corner_pts:
                new_corner = np.array([[corner[0]], [corner[1]], [corner[2]], [1]])

                local_pt = intrinsic_matrix @ extrinsic_matrix @ new_corner
                scaling = local_pt[2][0]
                local_pt /= scaling

                proj_x_pts.append(round(local_pt[0][0]))
                proj_y_pts.append(round(local_pt[1][0]))
            
            x_min = min(proj_x_pts)
            y_min = min(proj_y_pts)
            x_max = max(proj_x_pts)
            y_max = max(proj_y_pts)

            temp_bbox = [x_min, y_min, x_max, y_max]

            cnt += 1

            cv2.rectangle(self.image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 5)
        
            flag = 0

            for cam_bbox in self.cam_bbox_list:

                if (iou(cam_bbox, temp_bbox) > 0.2):
                    flag += 1
            
            if flag != 0:
                after_fusion_box_list.append(lidar_bbox)
        

        boxes.boxes = after_fusion_box_list

        self.fusion_pub.publish(boxes)

        cv2.imshow('Display', self.image)
        cv2.waitKey(1)

if __name__ == "__main__":
    
    if not rospy.is_shutdown():
        fusion()
        rospy.spin()
