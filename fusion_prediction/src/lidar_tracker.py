#!/usr/bin/env python

"""
    SORT: A Simple, Online and Realtime Tracker
    Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai
    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.
    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.
    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""
from __future__ import print_function

import numpy as np
from skimage import io

import time
import argparse
from filterpy.kalman import KalmanFilter
from shapely.geometry import Polygon

import rospy
from jsk_recognition_msgs.msg import BoundingBox, BoundingBoxArray
from visualization_msgs.msg import Marker, MarkerArray

import sys
import signal
import math

def signal_handler(signal, frame): # ctrl + c -> exit program
        print('You pressed Ctrl+C!')
        sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)


try:
  from numba import jit
except:
  def jit(func):
    return func
np.random.seed(0)

def is_none(bbox):
  
  if bbox.pose.position.x == None:
    return True
  if bbox.pose.position.y == None:
    return True
  if bbox.pose.position.z == None:
    return True
  if bbox.dimensions.x == None:
    return True
  if bbox.dimensions.y == None:
    return True
  if bbox.dimensions.z == None:
    return True
  if bbox.pose.orientation.x == None:
    return True
  if bbox.pose.orientation.y == None:
    return True
  if bbox.pose.orientation.z == None:
    return True
  if bbox.pose.orientation.w == None:
    return True
  
  return False


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

def linear_assignment(cost_matrix):
  try:
    import lap
    _, x, y = lap.lapjv(cost_matrix, extend_cost=True)
    return np.array([[y[i],i] for i in x if i >= 0]) #
  except ImportError:
    from scipy.optimize import linear_sum_assignment
    x, y = linear_sum_assignment(cost_matrix)
    return np.array(list(zip(x, y)))

def bbox2globalBEV(bbox):
  """
      Convert Object local coordinate into global coordinate.
      
      Output is a matrix of four points with global 2D BEV coordinates.
  """
  x_centroid = bbox.pose.position.x
  y_centroid = bbox.pose.position.y
  z_centroid = bbox.pose.position.z

  yaw = quaternion2euluer(bbox.pose.orientation.x, bbox.pose.orientation.y, bbox.pose.orientation.z, bbox.pose.orientation.w)
  
  matrix = np.array([[-math.cos(math.radians(yaw)), -math.sin(math.radians(yaw))], [math.sin(math.radians(yaw)), -math.cos(math.radians(yaw))]])

  width = bbox.dimensions.x
  length = bbox.dimensions.y
  height = bbox.dimensions.z

  local_pts = np.array([[-width/2, -length/2], [width/2, -length/2], [width/2, length/2], [-width/2, length/2]])

  rotated_corners = local_pts.dot(matrix)
  rotated_corners = rotated_corners + [x_centroid, y_centroid]

  return rotated_corners

def iou(bbox1, bbox2):
  """
  Computes IUO between two bboxes in the form [x1,y1,x2,y2]
  """

  rotated_coords1 = bbox2globalBEV(bbox1)
  rotated_coords2 = bbox2globalBEV(bbox2)

  first_bbox = Polygon([(rotated_coords1[0][0], rotated_coords1[0][1]), (rotated_coords1[1][0], rotated_coords1[1][1]), (rotated_coords1[2][0], rotated_coords1[2][1]), (rotated_coords1[3][0], rotated_coords1[3][1])])
  second_bbox = Polygon([(rotated_coords2[0][0], rotated_coords2[0][1]), (rotated_coords2[1][0], rotated_coords2[1][1]), (rotated_coords2[2][0], rotated_coords2[2][1]), (rotated_coords2[3][0], rotated_coords2[3][1])])

  intersection_area = first_bbox.intersection(second_bbox).area

  union_area = first_bbox.union(second_bbox).area

  result = intersection_area / union_area

  return result

def check_in_another_bbox(bbox, another_bbox, intersection_pts):
  
  rotated_coords = bbox2globalBEV(bbox)

  yaw = quaternion2euluer(another_bbox.pose.orientation.x, another_bbox.pose.orientation.y, another_bbox.pose.orientation.z, another_bbox.pose.orientation.w)

  R = np.array([[-math.cos(math.radians(yaw)), -math.sin(math.radians(yaw))], [math.sin(math.radians(yaw)), -math.cos(math.radians(yaw))]])
  inv_R = np.linalg.inv(R)

  x_centroid = another_bbox.pose.position.x
  y_centroid = another_bbox.pose.position.y

  w = another_bbox.dimensions.x
  l = another_bbox.dimensions.y
  
  t = np.array([x_centroid, y_centroid])

  converted_rotated_coords = (rotated_coords - t).dot(inv_R)

  for point in converted_rotated_coords:
     
     if point[0] >= -w/2 and point[0] <= w/2 and point[1] >= -l/2 and point[1] <= l/2:
        if point.tolist() not in intersection_pts:
          intersection_pts.append(point.tolist())

def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  w = bbox.dimensions.x
  l = bbox.dimensions.y
  x = bbox.pose.position.x
  y = bbox.pose.position.y
  s = w * l    #scale is just area
  r = w / float(l)
  z_orient = bbox.pose.orientation.z
  w_orient = bbox.pose.orientation.w
  return np.array([x, y, s, r, z_orient, w_orient]).reshape((6, 1))


def convert_x_to_bbox(x, z_orient, w_orient):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2] * x[3])
  l = x[2] / w
  box = BoundingBox()
  box.pose.position.x = x[0][0]
  box.pose.position.y = x[1][0]
  box.pose.position.z = -0.71
  box.dimensions.x = w[0]
  box.dimensions.y = l[0]
  box.dimensions.z = 1.43
  box.pose.orientation.x = 0.0
  box.pose.orientation.y = 0.0
  box.pose.orientation.z = z_orient
  box.pose.orientation.w = w_orient

  return box

class KalmanBoxTracker(object):
  """
  This class represents the internal state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    self.kf = KalmanFilter(dim_x=7, dim_z=4) 
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    self.kf.x[:4] = convert_bbox_to_z(bbox)[:4]
    self.z_orient = convert_bbox_to_z(bbox)[4]
    self.w_orient = convert_bbox_to_z(bbox)[5]
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0


  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox)[:4])
    self.z_orient = convert_bbox_to_z(bbox)[4]
    self.w_orient = convert_bbox_to_z(bbox)[5]


  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x, self.z_orient, self.w_orient))
    return self.history[-1]


  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x, self.z_orient, self.w_orient)


def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.1):
  """
  Assigns detections to tracked object (both represented as bounding boxes)
  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  if(len(trackers.boxes)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers.boxes)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers.boxes):
      iou_matrix[d,t] = iou(det,trk)

  if min(iou_matrix.shape) > 0:
    a = (iou_matrix > iou_threshold).astype(np.int32)
    if a.sum(1).max() == 1 and a.sum(0).max() == 1:
        matched_indices = np.stack(np.where(a), axis=1)
    else:
      matched_indices = linear_assignment(-iou_matrix)
  else:
    matched_indices = np.empty(shape=(0,2))

  unmatched_detections = []
  for d, det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t, trk in enumerate(trackers.boxes):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0], m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


class Sort(object):
  def __init__(self, max_age=5, min_hits=1):
    rospy.init_node('sort', anonymous=True)
    self.subb = rospy.Subscriber('display/detections', BoundingBoxArray, self.boxcallback)
    self.pubb = rospy.Publisher('fusion/tracks', BoundingBoxArray, queue_size=50)
    self.marker_pub = rospy.Publisher('text_marker', MarkerArray, queue_size=50)
    self.iou_threshold = rospy.get_param("~tracker_iou_threshold")

    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    #self.colours = np.random.rand(32, 3) #used only for display

    self.img_in = 0
    self.bbox_checkin = 0

  def boxcallback(self, msg):
    
    empty_box = BoundingBox()
    empty_box.pose.position.x = None
    empty_box.pose.position.y = None
    empty_box.pose.position.z = None
    empty_box.dimensions.x = None
    empty_box.dimensions.y = None
    empty_box.dimensions.z = None
    empty_box.pose.orientation.x = None
    empty_box.pose.orientation.y = None
    empty_box.pose.orientation.z = None
    empty_box.pose.orientation.w = None

    dets = []
    for i in range(len(msg.boxes)):
        bbox = msg.boxes[i]
        dets.append(bbox)
    self.dets = dets
    self.bbox_checkin=1

    start_time = time.time()
    if self.bbox_checkin==1:
        trackers = self.update(self.dets)
        self.bbox_checkin=0
    else:
        trackers = self.update(empty_box)
      
    final_boxes = BoundingBoxArray()
    final_boxes.header = msg.header
    final_markers = MarkerArray()

    tmp_cur_time = rospy.Time.now()

    for d in range(len(trackers)):
        if trackers[d].pose.position.x is not None:
          box = BoundingBox()
          box.header.frame_id = msg.header.frame_id
          box.header.stamp = rospy.Time.now()
        
          box.pose.position.x = float(trackers[d].pose.position.x)
          box.pose.position.y = float(trackers[d].pose.position.y)
          box.pose.position.z = float(trackers[d].pose.position.z)
          box.dimensions.x = float(trackers[d].dimensions.x)
          box.dimensions.y = float(trackers[d].dimensions.y)
          box.dimensions.z = float(trackers[d].dimensions.z)
          box.pose.orientation.x = float(trackers[d].pose.orientation.x)
          box.pose.orientation.y = float(trackers[d].pose.orientation.y)
          box.pose.orientation.z = float(trackers[d].pose.orientation.z)
          box.pose.orientation.w = float(trackers[d].pose.orientation.w)
        
          box.label = trackers[d].label

          result_str = "ID : " + str(box.label)

          marker = Marker()
          marker.header.frame_id = msg.header.frame_id
          marker.header.stamp = tmp_cur_time
          marker.type = marker.TEXT_VIEW_FACING
          marker.id = d+1
          marker.action = marker.ADD
          marker.scale.z = 2
          marker.color.r = 1.0
          marker.color.g = 0.0
          marker.color.b = 0.0
          marker.color.a = 1.0
          marker.pose.position.x = trackers[d].pose.position.x - 1
          marker.pose.position.y = trackers[d].pose.position.y
          marker.pose.position.z = 1
          marker.text = result_str

          final_markers.markers.append(marker)
          final_boxes.boxes.append(box)
        
    cycle_time = time.time() - start_time
    if len(final_boxes.boxes)>0: #prevent empty box
        final_boxes.header.stamp = rospy.Time.now()
        self.pubb.publish(final_boxes)
        self.marker_pub.publish(final_markers)
        # print(cycle_time)
    return

  def update(self, dets=BoundingBoxArray()):
    """
    Params:
      dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
    Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
    Returns the a similar array, where the last column is the object ID.
    NOTE: The number of objects returned may differ from the number of detections provided.
    """
    self.frame_count += 1
    # get predicted locations from existing trackers.
    trks = np.zeros((len(self.trackers), 5))
    empty_boxes = BoundingBoxArray()

    empty_box = BoundingBox()
    empty_box.pose.position.x = None
    empty_box.pose.position.y = None
    empty_box.pose.position.z = None
    empty_box.dimensions.x = None
    empty_box.dimensions.y = None
    empty_box.dimensions.z = None
    empty_box.pose.orientation.x = None
    empty_box.pose.orientation.y = None
    empty_box.pose.orientation.z = None
    empty_box.pose.orientation.w = None

    empty_boxes.boxes.append(empty_box)

    trks = BoundingBoxArray()

    for i in range(len(self.trackers)):
       trks.boxes.append(empty_box)
    
    to_del = []
    ret = []

    for t, trk in enumerate(trks.boxes):
      pos = self.trackers[t].predict()
      # trk = pos
      trks.boxes[t] = pos
      if is_none(pos) == True:
        to_del.append(t)
    # trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks,self.iou_threshold)

    # update matched trackers with assigned detections
    for m in matched:
      self.trackers[m[1]].update(dets[m[0]])

    # create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i])
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()
        if (trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits):
          d.label = trk.id+1
          ret.append(d) # +1 as MOT benchmark requires positive
        i -= 1
        # remove dead tracklet
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    if(len(ret)>0):
      return ret
    return empty_boxes.boxes

if __name__ == '__main__':
    colours = np.random.rand(32, 3) #used only for display
    mot_tracker = Sort(max_age=200, min_hits=1) #create instance of the SORT tracker
    rospy.spin()