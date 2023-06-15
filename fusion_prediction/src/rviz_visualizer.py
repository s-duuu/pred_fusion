#!/usr/bin/env python3
# -*- coding:utf-8 -*-

#######################################################################
# Copyright (C) 2022 EunGi Han(winterbloooom) (winterbloooom@gmail.com)
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program. If not, see <http://www.gnu.org/licenses/>

"""visualization module using Rviz (2D)"""


from geometry_msgs.msg import Point, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray


def basic_setting(name, id, color_r, color_g, color_b, color_a=255):
    """make Marker object with basic settings

    Args:
        * name (str) : Marker name. It can share same name with others. (마커의 이름이며, 다른 마커와 중복될 수 있음.)
        * id (int) : Marker id. It should be unique. (마커의 아이디이며, 각 마커마다 고유한 숫자가 부여되어야 함.)
        * color_r, color_g, color_b (int) : Marker color in RGB format (0 ~ 255) (마커의 RGB 색을 0~255 범위로 표현함.)
        * color_a (int) : Transparency (0 ~ 255) (마커의 투명도를 0~255 범위로 표현함.)

    Returns:
        marker (Marker): Marker object with basic settings (기본 설정을 담은 마커를 만들어 반환함.)

    Notes:
        * set frame_id, namespace, id, action, color, orientation (기본 설정에는 프레임 아이디, 이름, 마커 아이디, 동작, 색, 위치를 포함함.)
        * ColorRGBA는 0~1사이 값을 사용하므로 편의상 0~255 단위로 입력받아 여기서 255로 나누어줌
    """
    marker = Marker()
    marker.header.frame_id = "LIRS00"
    marker.ns = name
    marker.id = id
    marker.action = Marker.ADD
    marker.color = ColorRGBA(color_r / 255.0, color_g / 255.0, color_b / 255.0, color_a / 255.0)
    marker.pose.orientation.w = 1.0

    return marker


def del_mark(name, id):
    """delete existing marker if not necessary

    특정 ID를 가진 마커가 있을 때, 이를 삭제함.
    다음 rate에서 해당 아이디의 마커를 갱신할 것이 아니면 계속 화면 상에 남아있기 때문에 del_mark() 함수를 구현함.

    Args:
        * name (str) : Marker name (마커 이름)
        * id (int) : Marker id (마커 아이디)

    Returns:
        marker (Marker): Marker object to delete (삭제할 마커를 만들어 반환함.)
    """
    marker = Marker()
    marker.header.frame_id = "/map"
    marker.ns = name
    marker.id = id
    marker.action = Marker.DELETE

    return marker


def point_rviz(name, id, point, color_r=0, color_g=0, color_b=0, color_a=255, scale=0.1):
    """make one Point Marker(점 '하나'의 마커를 만들어 반환함)

    Args:
        name (str) : Marker name. It can share same name with others
        id (int) : Marker id. It should be unique.
        point (list) : point position in meter [x(float), y(float)] (미터 단위로 점의 위치를 지정. [x, y]식)
        color_r, color_g, color_b (int) : (default: 0) Marker color in RGB format (0 ~ 255)
        color_a (int) : (default: 255) Transparency (0 ~ 255)
        scale (float) : (default: 0.1) size of point in meter (미터 단위로 점의 크기를 지정.)

    Returns:
        marker (Marker) : Point Marker object
    """
    marker = basic_setting(name, id, color_r, color_g, color_b, color_a)
    marker.type = Marker.POINTS
    marker.scale = Vector3(scale, scale, 0)
    marker.points.append(Point(point[0], point[1], 0))

    return marker


def points_rviz(name, id, points, color_r=0, color_g=0, color_b=0, color_a=255, scale=0.1):
    """make a set of Point Marker(점 '여러 개'의 마커를 만들어 반환함)

    Args:
        name (str) : Marker name. It can share same name with others
        id (int) : Marker id. It should be unique.
        points (list) : point positions in meter ([[x, y], [x, y], ...])
        color_r, color_g, color_b (int) : Marker color in RGB format (0 ~ 255)
        color_a (int) : Transparency (0 ~ 255)
        scale (float) : size of point in meter

    Returns:
        marker (Marker): Point Marker object
    """
    marker = basic_setting(name, id, color_r, color_g, color_b, color_a)
    marker.type = Marker.POINTS
    marker.scale = Vector3(scale, scale, 0)
    for point in points:
        marker.points.append(Point(point[0], point[1], 0))

    return marker


def arrow_rviz(
    name, id, tail, head, color_r=0, color_g=0, color_b=0, color_a=255, scale_x=0.2, scale_y=0.4
):
    """make a Arrow Marker (화살표 마커를 만들어 반환함)

    Args:
        name (str) : Marker name. It can share same name with others
        id (int) : Marker id. It should be unique.
        tail (list) : Arrow tail point position in meter. [x, y] format. (화살표 꼬리의 좌표 [x, y] 형태로)
        head (list) : Arrow head point position in meter. [x, y] format. (화살표 머리의 좌표 [x, y] 형태로)
        color_r, color_g, color_b (int) : Marker color in RGB format (0 ~ 255)
        color_a (int) : Transparency (0 ~ 255)
        scale_x, scale_y (float) : size of arrow in meter

    Returns:
        marker (Marker): Arrow Marker object
    """
    marker = basic_setting(name, id, color_r, color_g, color_b, color_a)
    marker.type = Marker.ARROW
    marker.scale = Vector3(scale_x, scale_y, 0)
    marker.points.append(Point(tail[0], tail[1], 0))
    marker.points.append(Point(head[0], head[1], 0))

    return marker


def text_rviz(name, id, position, text, scale=0.6):
    """make a Text Marker (텍스트 마커를 만들어 반환함)

    Args:
        name (str) : Marker name. It can share same name with others
        id (int) : Marker id. It should be unique.
        position (list) : position of text in meter, [x, y] format (텍스트 위치의 미터 단위의 [x, y] 좌표)
        text (str) : text to write (입력할 텍스트)
        scale (float) : size of text in meter (미터 단위의 텍스트 크기)

    Returns:
        marker (Marker): Point Marker object
    """
    marker = basic_setting(name, id, color_r=255, color_g=255, color_b=255)
    marker.type = Marker.TEXT_VIEW_FACING
    marker.pose.position = Point(position[0], position[0], 0)
    marker.scale.z = scale
    marker.text = text

    return marker


def linelist_rviz(name, id, lines, color_r=0, color_g=0, color_b=0, color_a=255, scale=0.05):
    """make a Line List Marker (라인 리스트 마커를 만들어 반환함)

    Args:
        name (str) : Marker name. It can share same name with others
        id (int) : Marker id. It should be unique.
        lines (list) : set of lines' each end positions. [[begin_x, begin_y], [end_x, end_y], [begin_x, begin_y], [end_x, end_y], ...]
        color_r, color_g, color_b (int) : Marker color in RGB format (0 ~ 255)
        color_a (int) : Transparency (0 ~ 255)
        scale (float) : thickness of Line List in meter

    Returns:
        marker (Marker): Line List Marker object
    """
    marker = basic_setting(name, id, color_r, color_g, color_b, color_a)
    marker.type = Marker.LINE_LIST
    marker.scale.x = scale
    for line in lines:
        marker.points.append(Point(line[0], line[1], 0))

    return marker


def cylinder_rviz(name, id, center, scale, color_r=0, color_g=0, color_b=0, color_a=150):
    """make a Cylinder Marker (실린더 마커를 만들어 반환함. 높이를 최소한으로 부여해 원을 그림.)

    Args:
        name (str) : Marker name. It can share same name with others
        id (int) : Marker id. It should be unique.
        center (list) : position of cylinder center in meter, [x, y] format (원 중심의 미터 단위의 [x, y] 좌표)
        scale (float) : diameter of cylinder (원의 미터 단위의 지름)
        color_r, color_g, color_b (int) : Marker color in RGB format (0 ~ 255)
        color_a (int) : Transparency (0 ~ 255)

    Returns:
        marker (Marker): Cylinder Marker object
    """
    marker = basic_setting(name, id, color_r, color_g, color_b, color_a)
    marker.type = Marker.CYLINDER
    marker.scale = Vector3(scale, scale, 0.01)
    marker.pose.position = Point(center[0], center[1], 0)

    return marker


def marker_array_rviz(markers):
    """make a MarkerArray object (여러 개의 Markers를 하나의 MarkerArray로 만들어 반환)

    Args:
        markers (list) : list of Marker objects. [marker, marker, ...]

    Returns:
        MarkerArray : MarkerArray object having input markers
    """
    marker_array = MarkerArray()
    for marker in markers:
        marker_array.markers.append(marker)

    return marker_array


def marker_array_append_rviz(marker_array, marker):
    """append one Marker object in exisiting MarkerArray object (기존의 MarkerArray에 특정 Marker를 추가함)

    Args:
        marker_array (MarkerArray) : MarkerArray object
        marker (Marker) : Marker objects to append

    Returns:
        MarkerArray : MarkerArray object
    """
    marker_array.markers.append(marker)

    return marker_array