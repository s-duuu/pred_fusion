#!/usr/bin/env python3

import rospy
from sensor_msgs.msg import PointCloud2
import sensor_msgs.point_cloud2 as pc2
from std_msgs.msg import Header
from pathlib import Path
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray

from time import time
import numpy as np
from pyquaternion import Quaternion

import numpy as np
import torch
import scipy.linalg as linalg
import sys
import os

sys.path.append('/home/heven/pred_ws/src/pred_fusion/PointPillars')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0] / "PointPillars"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

from model import PointPillars

sys.path.append('/home/heven/pred_ws/src/pred_fusion/OpenPCDet')

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0].parents[0].parents[0] / "OpenPCDet"
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative path

from pcdet.config import cfg, cfg_from_yaml_file
from pcdet.datasets import DatasetTemplate
from pcdet.utils import common_utils

def point_range_filter(pts, point_range=[0, -39.68, -3, 69.12, 39.68, 1]):
    '''
    data_dict: dict(pts, gt_bboxes_3d, gt_labels, gt_names, difficulty)
    point_range: [x1, y1, z1, x2, y2, z2]
    '''
    flag_x_low = pts[:, 0] > point_range[0]
    flag_y_low = pts[:, 1] > point_range[1]
    flag_z_low = pts[:, 2] > point_range[2]
    flag_x_high = pts[:, 0] < point_range[3]
    flag_y_high = pts[:, 1] < point_range[4]
    flag_z_high = pts[:, 2] < point_range[5]
    keep_mask = flag_x_low & flag_y_low & flag_z_low & flag_x_high & flag_y_high & flag_z_high
    pts = pts[keep_mask]
    return pts 

def quaternion2euluer(x, y, z, w):
    """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
    """
    import math
    t0 = +2.0 * (w * x + y * z)
    t1 = +1.0 - 2.0 * (x * x + y * y)
    roll_x = math.atan2(t0, t1)
    
    t2 = +2.0 * (w * y - z * x)
    t2 = +1.0 if t2 > +1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch_y = math.asin(t2)
    
    t3 = +2.0 * (w * z + x * y)
    t4 = +1.0 - 2.0 * (y * y + z * z)
    yaw_z = math.atan2(t3, t4)
    
    return math.degrees(yaw_z) # in degrees

class DemoDataset(DatasetTemplate):
    def __init__(self, dataset_cfg, class_names, training=True, root_path=None, logger=None, ext='.bin'):
        super().__init__(
            dataset_cfg=dataset_cfg, class_names=class_names, training=training, root_path=root_path, logger=logger
        )

class Pointpillars_ROS:
    def __init__(self):
        
        config_path, ckpt_path = self.init_ros()
        self.init_pointpillars(config_path, ckpt_path)


    def init_ros(self):
        """ Initialize ros parameters """
        rospy.init_node('pointpillars_ros_node', anonymous=True)
        config_path = rospy.get_param("~config_path")
        ckpt_path = rospy.get_param("~ckpt_path")
        lidar_topic = rospy.get_param("~input_lidar_topic")
        self.score_thresh = rospy.get_param("~score_threshold")
        self.sub_velo = rospy.Subscriber(lidar_topic, PointCloud2, self.lidar_callback, queue_size=1,  buff_size=2**12)
        self.pub_bbox = rospy.Publisher("/pillars/detections", BoundingBoxArray, queue_size=1)
        return config_path, ckpt_path


    def init_pointpillars(self, config_path, ckpt_path):
        """ Initialize second model """
        logger = common_utils.create_logger()
        logger.info('-----------------Quick Demo of Pointpillars-------------------------')
        cfg_from_yaml_file(config_path, cfg)
        
        self.demo_dataset = DemoDataset(
            dataset_cfg=cfg.DATA_CONFIG, class_names=cfg.CLASS_NAMES, training=False,
            logger=logger
        )
        CLASSES = {
        'Pedestrian': 0, 
        'Cyclist': 1, 
        'Car': 2
        }
        self.model = PointPillars(nclasses=len(CLASSES)).cuda()
        self.model.load_state_dict(torch.load(ckpt_path))
        
        self.model.cuda()
        self.model.eval() 


    def rotate_mat(self, axis, radian):
        rot_matrix = linalg.expm(np.cross(np.eye(3), axis / linalg.norm(axis) * radian))
        return rot_matrix


    def lidar_callback(self, msg):
        """ Captures pointcloud data and feed into second model for inference """

        np_p = np.frombuffer(msg.data, dtype=np.float32).reshape(-1, 4)
        np_p = point_range_filter(np_p)

        np_copy = np.copy(np_p)
        np_copy[:,3] = np_copy[:,3] / max(np_copy[:,3])

        pc_torch = torch.from_numpy(np_copy)

        with torch.no_grad():
            pc_torch = pc_torch.cuda()

        result_filter = self.model(batched_pts = [pc_torch], mode='test')[0]

        lidar_bboxes = result_filter['lidar_bboxes']
        labels, scores = result_filter['labels'], result_filter['scores']

        num_detections = len(lidar_bboxes)
        arr_bbox = BoundingBoxArray()
        for i in range(num_detections):
            bbox = BoundingBox()

            bbox.header.frame_id = msg.header.frame_id
            bbox.header.stamp = rospy.Time.now()
            bbox.pose.position.x = float(lidar_bboxes[i][0])
            bbox.pose.position.y = float(lidar_bboxes[i][1])
            bbox.pose.position.z = float(lidar_bboxes[i][2]) + float(lidar_bboxes[i][5]) / 2
            bbox.dimensions.x = float(lidar_bboxes[i][3])  # width
            bbox.dimensions.y = float(lidar_bboxes[i][4])  # length
            bbox.dimensions.z = float(lidar_bboxes[i][5])  # height
            q = Quaternion(axis=(0, 0, -1), radians=float(lidar_bboxes[i][6]))
            bbox.pose.orientation.x = q.x
            bbox.pose.orientation.y = q.y
            bbox.pose.orientation.z = q.z
            bbox.pose.orientation.w = q.w
            bbox.value = scores[i]
            bbox.label = labels[i]

            if scores[i] > self.score_thresh:
                arr_bbox.boxes.append(bbox)
        
        arr_bbox.header.frame_id = msg.header.frame_id
        arr_bbox.header.stamp = rospy.Time.now()
        
        self.pub_bbox.publish(arr_bbox)

if __name__ == '__main__':
    sec = Pointpillars_ROS()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        del sec
        print("Shutting down")
