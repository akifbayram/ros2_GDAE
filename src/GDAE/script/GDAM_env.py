# GDAM_env.py

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, qos_profile_sensor_data

from numpy import inf
import subprocess
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Twist
from pyquaternion import Quaternion
from collections import deque
import numpy as np
import math
from statistics import stdev

import tf_transformations
import tf2_ros
from tf2_ros import TransformException
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose
from nav_msgs.msg import Odometry, Path
import os


def rotateneg(point, angle):
    x, y = point
    xx = x * math.cos(angle) + y * math.sin(angle)
    yy = -x * math.sin(angle) + y * math.cos(angle)
    return xx, yy


def rotatepos(point, angle):
    x, y = point
    xx = x * math.cos(angle) - y * math.sin(angle)
    yy = x * math.sin(angle) + y * math.cos(angle)
    return xx, yy


def calcqxqy(dist, angl, ang):
    angl = math.radians(angl)
    angle = angl + ang
    if angle > np.pi:
        angle = angle - 2 * np.pi
    if angle < -np.pi:
        angle = angle + 2 * np.pi
    if angle > 0:
        qx, qy = rotatepos([dist, 0], angle)
    else:
        qx, qy = rotateneg([dist, 0], -angle)
    return qx, qy


class ImplementEnv(Node):
    def __init__(self, args):

        super().__init__('gym')

        self.countZero = 0
        self.count_turn = -100

        self.node_vicinity = args.node_vicinity
        self.del_node_vicinity = args.deleted_node_vicinity
        self.min_in = args.min_in
        self.side_min_in = args.side_min_in
        self.del_nodes_range = args.delete_nodes_range
        # launchfile = args.launchfile
        self.accelPos_low = args.acceleration_low
        self.accelPos_high = args.acceleration_high
        self.accelNeg_low = args.deceleration_low
        self.accelNeg_high = args.deceleration_high
        self.angPos = args.angular_acceleration
        self.original_goal_x = args.x
        self.original_goal_y = args.y
        nr_nodes = args.nr_of_nodes
        nr_closed_nodes = args.nr_of_closed_nodes
        nr_deleted_nodes = args.nr_of_deleted_nodes
        self.update_rate = args.update_rate
        self.remove_rate = args.remove_rate
        self.stddev_threshold = args.stddev_threshold
        self.freeze_rate = args.freeze_rate

        self.accelNeg = 0

        self.angle = 0
        self.odomX = 0.0
        self.odomY = 0.0
        self.linearLast = 0.0
        self.angularLast = 0.0

        self.LaserData = None
        # Removed LaserDataTop subscription as it's not available
        self.OdomData = None

        self.lock = 0

        self.PathData = None

        self.last_statesX = deque(maxlen=self.freeze_rate)
        self.last_statesY = deque(maxlen=self.freeze_rate)
        self.last_statesX.append(0.0)
        self.last_statesY.append(0.0)

        self.last_states = False

        self.global_goal_x = self.original_goal_x
        self.global_goal_y = self.original_goal_y

        self.nodes = deque(maxlen=nr_nodes)
        self.nodes.append([4.5, 0.0, 4.5, 0.0, 0])
        self.nodes.append([self.global_goal_x, self.global_goal_y, self.global_goal_x, self.global_goal_y, 0])

        self.closed_nodes = deque(maxlen=nr_closed_nodes)
        self.closed_nodes.append([0.0, 0.0])
        self.closed_nodes_rotated = deque(maxlen=nr_closed_nodes)

        self.map_nodes = deque(maxlen=400)

        self.deleted_nodes = deque(maxlen=nr_deleted_nodes)
        self.deleted_nodes_rotated = deque(maxlen=nr_deleted_nodes)

        self.goalX = self.nodes[0][2]
        self.goalY = self.nodes[0][3]

        self.g_node = 0

        # Commented out the launch file related code as per user request
        """
        # ROS 2 does not use ROS_PORT_SIM. Launch files are handled externally.
        if launchfile.startswith("/"):
            fullpath = launchfile
        else:
            fullpath = os.path.join(os.path.dirname(__file__), "assets", "launch", launchfile)
        if not os.path.exists(fullpath):
            raise IOError("File " + fullpath + " does not exist")

        # Start the launch file using subprocess (ensure it is compatible with ROS 2)
        subprocess.Popen(["ros2", "launch", fullpath])
        """

        qos = QoSProfile(depth=10)
        sensor_qos = qos_profile_sensor_data

        # Updated publishers to match available topics
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)
        # Removed 'RosAria' publisher since it's not in the topic list
        topic_nodes_pub = 'vis_mark_array_node'
        self.nodes_pub = self.create_publisher(Marker, topic_nodes_pub, qos)
        topic_nodes_pub_closed = 'vis_mark_array_node_closed'
        self.nodes_pub_closed = self.create_publisher(Marker, topic_nodes_pub_closed, qos)
        topic_map_nodes = 'vis_mark_array_map_nodes'
        self.map_nodes_viz = self.create_publisher(Marker, topic_map_nodes, qos)
        topic = 'vis_mark_array'
        self.publisher = self.create_publisher(MarkerArray, topic, qos)
        self.global_goal_publisher = self.create_publisher(MarkerArray, 'global_goal_publisher', qos)

        # Updated subscriptions to match available topics
        self.navLaser = self.create_subscription(LaserScan, '/scan', self.Laser_callback, sensor_qos)
        # Removed '/rpTop/scan' subscription as it's not available
        self.navOdom = self.create_subscription(Odometry, '/odom', self.Odom_callback, qos)
        self.path = self.create_subscription(Path, '/plan', self.path_callback, qos)

        # Updated action client to use Nav2's NavigateToPose
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        while not self.client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for action server navigate_to_pose...')

        # Updated TF listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def Laser_callback(self, l):
        self.LaserData = l

    def Odom_callback(self, o):
        self.OdomData = o

    def path_callback(self, p):
        self.PathData = p

    def step(self, act):
        rplidar = None
        # Removed rplidarTop as it's not available
        self.map_nodes.clear()
        while rplidar is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            rplidar = self.LaserData
            if rplidar is None:
                self.get_logger().info("Waiting for laser scans...")

        dataOdom = None
        while dataOdom is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            dataOdom = self.OdomData
            if dataOdom is None:
                self.get_logger().info("Waiting for odometry data...")

        # Dynamically handle the number of laser scan points
        num_points = len(rplidar.ranges)
        if num_points == 0:
            self.get_logger().warn("Received empty laser scan.")
            return np.array([]), [0, 0]

        # Example: Divide the laser scan into 19 segments as in the original code
        num_segments = 19
        segment_size = num_points // num_segments
        laser_state = []
        for i in range(num_segments):
            start_idx = i * segment_size
            end_idx = (i + 1) * segment_size if i < num_segments - 1 else num_points
            segment = rplidar.ranges[start_idx:end_idx]
            # Filter out 'inf' and set a maximum range
            segment = [min(r, 10.0) for r in segment if not math.isinf(r)]
            if segment:
                laser_state.append(min(segment))
            else:
                laser_state.append(10.0)  # Default max value if segment is empty

        laser_state = np.array(laser_state) / 10.0  # Normalize

        # Laser collision check
        col, colleft, colright, minleft, minright = self.laser_check(rplidar.ranges)

        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        quaternion = (
            dataOdom.pose.pose.orientation.x,
            dataOdom.pose.pose.orientation.y,
            dataOdom.pose.pose.orientation.z,
            dataOdom.pose.pose.orientation.w)
        euler = tf_transformations.euler_from_quaternion(quaternion)
        self.angle = round(euler[2], 4)

        try:
            transform = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())
            trans = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            rot = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
        except TransformException as ex:
            self.get_logger().info(f'Could not transform odom to map: {ex}')
            return laser_state, [0, 0]

        local_x = self.original_goal_x - trans[0]
        local_y = self.original_goal_y - trans[1]
        eul = tf_transformations.euler_from_quaternion(rot)
        q8c = Quaternion(axis=[0.0, 0.0, 1.0], radians=-eul[2])
        q9c = Quaternion(axis=[0.0, 0.0, 1.0], radians=eul[2])
        p_tmp = q8c.rotate([local_x, local_y, 0])
        self.global_goal_x = p_tmp[0]
        self.global_goal_y = p_tmp[1]

        global_goal_distance = math.sqrt((self.odomX - self.global_goal_x)**2 + (self.odomY - self.global_goal_y)**2)
        if global_goal_distance < 1.5:
            self.get_logger().info("Arrived at the goal")
            # Stop the robot
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
            return laser_state, [0, 0]

        self.new_nodes(laser_state, self.odomX, self.odomY, self.angle, trans, q9c)
        self.free_space_nodes(laser_state, self.odomX, self.odomY, self.angle, trans, q9c)
        self.infinite_nodes(laser_state, self.odomX, self.odomY, self.angle, trans, q9c)

        if not self.nodes:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.1
            self.vel_pub.publish(vel_cmd)
            self.get_logger().info("Looking for nodes")
            return laser_state, [0, 0]

        for i_node in range(len(self.nodes)):
            if self.nodes[i_node][4] == 0:
                node = [self.nodes[i_node][0], self.nodes[i_node][1]]
                local_x = node[0] - trans[0]
                local_y = node[1] - trans[1]
                p_tmp = q8c.rotate([local_x, local_y, 0])
                self.nodes[i_node][2] = p_tmp[0]
                self.nodes[i_node][3] = p_tmp[1]

        self.closed_nodes_rotated.clear()
        for i_node in range(len(self.closed_nodes)):
            node = self.closed_nodes[i_node]
            local_x = node[0] - trans[0]
            local_y = node[1] - trans[1]
            p_tmp = q8c.rotate([local_x, local_y, 0])
            self.closed_nodes_rotated.append([p_tmp[0], p_tmp[1]])

        self.deleted_nodes_rotated.clear()
        for i_node in range(len(self.deleted_nodes)):
            node = self.deleted_nodes[i_node]
            local_x = node[0] - trans[0]
            local_y = node[1] - trans[1]
            p_tmp = q8c.rotate([local_x, local_y, 0])
            self.deleted_nodes_rotated.append([p_tmp[0], p_tmp[1]])

        if not self.last_states:
            self.last_states = self.freeze(self.odomX, self.odomY)
            self.count_turn = 8

        self.goalX = self.nodes[self.g_node][2]
        self.goalY = self.nodes[self.g_node][3]

        Dist = math.sqrt((self.odomX - self.goalX) ** 2 + (self.odomY - self.goalY) ** 2)

        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY

        dot = skewX * 1.0 + skewY * 0.0
        mag1 = math.sqrt(skewX ** 2 + skewY ** 2)
        mag2 = 1.0  # sqrt(1^2 + 0^2) = 1
        beta = math.acos(dot / (mag1 * mag2)) if mag1 != 0 else 0.0

        if skewY < 0:
            beta = -beta

        beta2 = beta - self.angle

        # Normalize angle to [-pi, pi]
        if beta2 > np.pi:
            beta2 -= 2 * np.pi
        if beta2 < -np.pi:
            beta2 += 2 * np.pi

        linear = act[0]
        angular = act[1]

        # Recover velocities based on collision
        linear, angular = self.recover(linear, angular, minleft, minright, colleft, colright, col)

        vel_cmd = Twist()
        vel_cmd.linear.x = float(linear)      # Ensure float type
        vel_cmd.angular.z = float(angular)   # Ensure float type

        if abs(angular) < 0.3:
            if angular > 0.0:
                vel_cmd.angular.z = (angular ** 2) / 0.3
            else:
                vel_cmd.angular.z = -(angular ** 2) / 0.3
        else:
            vel_cmd.angular.z = float(angular)  # Ensure float type

        self.vel_pub.publish(vel_cmd)

        self.linearLast = linear
        self.angularLast = angular

        # Visualization markers
        sphere_list = Marker()
        sphere_list.header.frame_id = "odom"
        sphere_list.type = Marker.SPHERE_LIST
        sphere_list.action = Marker.ADD
        sphere_list.scale.x = 0.3
        sphere_list.scale.y = 0.1
        sphere_list.scale.z = 0.01
        sphere_list.color.a = 1.0
        sphere_list.color.r = 0.0
        sphere_list.color.g = 0.0
        sphere_list.color.b = 1.0
        sphere_list.pose.orientation.w = 1.0
        for i_node in range(len(self.nodes)):
            p = Point()
            node = self.nodes[i_node]
            p.x = float(node[2])  # Ensure float type
            p.y = float(node[3])  # Ensure float type
            p.z = 0.0             # Ensure float type
            sphere_list.points.append(p)
        self.nodes_pub.publish(sphere_list)

        closed_list = Marker()
        closed_list.header.frame_id = "odom"
        closed_list.type = Marker.SPHERE_LIST
        closed_list.action = Marker.ADD
        closed_list.scale.x = 0.15
        closed_list.scale.y = 0.1
        closed_list.scale.z = 0.01
        closed_list.color.a = 1.0
        closed_list.color.r = 1.0
        closed_list.color.g = 0.5
        closed_list.color.b = 0.5
        closed_list.pose.orientation.w = 1.0
        for i_node in range(len(self.closed_nodes_rotated)):
            node = self.closed_nodes_rotated[i_node]
            p = Point()
            p.x = float(node[0])  # Ensure float type
            p.y = float(node[1])  # Ensure float type
            p.z = 0.0             # Ensure float type
            closed_list.points.append(p)
        self.nodes_pub_closed.publish(closed_list)

        map_list = Marker()
        map_list.header.frame_id = "odom"
        map_list.type = Marker.SPHERE_LIST
        map_list.action = Marker.ADD
        map_list.scale.x = 0.25
        map_list.scale.y = 0.1
        map_list.scale.z = 0.01
        map_list.color.a = 1.0
        map_list.color.r = 1.0
        map_list.color.g = 0.0
        map_list.color.b = 0.0
        map_list.pose.orientation.w = 1.0

        for i_node in range(len(self.map_nodes)):
            p = Point()
            node = self.map_nodes[i_node]
            p.x = float(node[0])  # Ensure float type
            p.y = float(node[1])  # Ensure float type
            p.z = 0.0             # Ensure float type
            map_list.points.append(p)
        self.map_nodes_viz.publish(map_list)

        markerArray = MarkerArray()
        marker = Marker()
        marker.header.frame_id = "odom"
        marker.type = Marker.CYLINDER
        marker.action = Marker.ADD
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.01
        marker.color.a = 1.0
        marker.color.r = 0.0
        marker.color.g = 1.0
        marker.color.b = 0.0
        marker.pose.orientation.w = 1.0
        marker.pose.position.x = float(self.goalX)  # Ensure float type
        marker.pose.position.y = float(self.goalY)  # Ensure float type
        marker.pose.position.z = 0.0             # Ensure float type

        markerArray.markers.append(marker)
        self.publisher.publish(markerArray)

        markerArrayGoal = MarkerArray()
        markerGoal = Marker()
        markerGoal.header.frame_id = "odom"
        markerGoal.type = Marker.CYLINDER
        markerGoal.action = Marker.ADD
        markerGoal.scale.x = 0.6
        markerGoal.scale.y = 0.6
        markerGoal.scale.z = 0.01
        markerGoal.color.a = 1.0
        markerGoal.color.r = 0.1
        markerGoal.color.g = 0.9
        markerGoal.color.b = 0.0
        markerGoal.pose.orientation.w = 1.0
        markerGoal.pose.position.x = float(self.global_goal_x)  # Ensure float type
        markerGoal.pose.position.y = float(self.global_goal_y)  # Ensure float type
        markerGoal.pose.position.z = 0.0                   # Ensure float type

        markerArrayGoal.markers.append(markerGoal)
        self.global_goal_publisher.publish(markerArrayGoal)

        if Dist < 1.0:
            self.change_goal()
            self.countZero = 0

        if Dist > 5.0:
            p_data = self.PathData

            try:
                path_len = len(p_data.poses) - 1
            except:
                path_len = 0

            c_p = 0
            while c_p <= path_len and path_len > 0:
                plan_x = p_data.poses[c_p].pose.position.x
                plan_y = p_data.poses[c_p].pose.position.y
                node = [plan_x, plan_y]
                local_x = node[0] - trans[0]
                local_y = node[1] - trans[1]
                p_tmp = q8c.rotate([local_x, local_y, 0])

                d = math.sqrt((self.odomX - p_tmp[0])**2 + (self.odomY - p_tmp[1])**2)
                if d > 4.0 or (c_p == path_len and d > 1.5):
                    f = True
                    for j in range(len(self.nodes)):
                        check_d = math.sqrt(
                            (self.nodes[j][2] - p_tmp[0])**2 + (self.nodes[j][3] - p_tmp[1])**2)
                        if check_d < 1.0:
                            self.g_node = j
                            f = False
                            break
                    if f:
                        self.nodes.append([p_data.poses[c_p].pose.position.x, p_data.poses[c_p].pose.position.y,
                                           p_data.poses[c_p].pose.position.x, p_data.poses[c_p].pose.position.y, 0])
                        self.g_node = len(self.nodes) - 1
                    break
                c_p += 1

        self.check_pos(self.odomX, self.odomY)

        if self.countZero % 30 == 0:
            f = True
            for j in range(len(self.closed_nodes_rotated)):
                d = math.sqrt(
                    (self.closed_nodes_rotated[j][0] - self.odomX)**2 + (self.closed_nodes_rotated[j][1] - self.odomY)**2)
                if d < 0.85:
                    f = False
                    break
            if f:
                node = [self.odomX, self.odomY]
                p_tmp = q9c.rotate([node[0], node[1], 0])
                self.closed_nodes.append([p_tmp[0] + trans[0], p_tmp[1] + trans[1]])

        if self.countZero % self.update_rate == 0:
            self.check_goal()

        if self.countZero > self.remove_rate:
            self.change_goal()
            self.countZero = 0

        self.countZero += 1
        Dist = min(5.0, Dist)
        toGoal = [Dist / 10.0, (beta2 + np.pi) / (np.pi * 2)]
        return laser_state, toGoal

    def recover(self, linear, angular, minleft, minright, colleft=False, colright=False, col=False):

        if colleft and colright:
            angular = 0.0
            linear = 0.3

        if col:
            if self.lock == 0:
                if minright - minleft > 0:
                    self.lock = 1
                else:
                    self.lock = -1
            angular = self.lock
            linear = 0.0
        else:
            self.lock = 0

        if self.last_states:
            if col:
                angular = 0.7
                linear = 0.0
            else:
                angular = 0.0
                linear = 0.35
                self.count_turn -= 1
            if self.count_turn < 0:
                self.last_states = False

        if linear > self.linearLast:
            self.accelNeg = 0
            if self.linearLast > 0.25:
                linear = min(self.linearLast + self.accelPos_low, linear)
            else:
                linear = min(self.linearLast + self.accelPos_high, linear)
        if linear < self.linearLast:
            if self.linearLast > 0.25:
                self.accelNeg += self.accelNeg_low
                linear = max(self.linearLast - self.accelNeg, linear)
            else:
                self.accelNeg += self.accelNeg_high
                linear = max(self.linearLast - self.accelNeg, linear)

        if self.angularLast < angular:
            angular = min(self.angularLast + self.angPos, angular)
        if self.angularLast > angular:
            angular = max(self.angularLast - self.angPos, angular)

        return linear, angular

    def heuristic(self, odomX, odomY, candidateX, candidateY):
        to_goal2 = True
        gX = self.global_goal_x
        gY = self.global_goal_y
        d = 0

        if to_goal2:
            d1 = math.sqrt((candidateX - odomX) ** 2 + (candidateY - odomY) ** 2)
            d2 = math.sqrt((candidateX - gX) ** 2 + (candidateY - gY) ** 2)
            if 5 < d1 < 10:
                d1 = 5
            if d1 < 5:
                d1 = 0
            d = d1 + d2
        return d

    def change_goal(self):
        if self.nodes:
            try:
                self.nodes.remove(self.nodes[self.g_node])
                self.check_goal()
            except IndexError:
                self.get_logger().warn("Attempted to remove a non-existing node.")

    def check_goal(self):
        if not self.nodes:
            self.get_logger().info("No nodes available to set as goal.")
            return

        min_d = self.heuristic(self.odomX, self.odomY, self.nodes[0][2], self.nodes[0][3])
        node_out = 0

        for i in range(len(self.nodes)):
            d = self.heuristic(self.odomX, self.odomY, self.nodes[i][2], self.nodes[i][3])
            if d < min_d:
                min_d = d
                node_out = i

        self.g_node = node_out
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'  # Ensure 'map' frame exists
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = self.nodes[self.g_node][2]
        goal_msg.pose.pose.position.y = self.nodes[self.g_node][3]
        goal_msg.pose.pose.orientation.w = 1.0
        self.client.send_goal_async(goal_msg)

    def new_nodes(self, laser, odomX, odomY, ang, trans, q9c):
        for i in range(1, len(laser)):
            if len(self.nodes) > 0 and laser[i] < 7.5:
                dist = laser[i]
                angl = i / 4 - 90
                qx, qy = calcqxqy(dist, angl, ang)

                self.map_nodes.append([qx + odomX, qy + odomY])

                for j in range(len(self.nodes) - 1, -1, -1):
                    d = math.sqrt(
                        (self.nodes[j][2] - qx - odomX) ** 2 + (self.nodes[j][3] - qy - odomY) ** 2)
                    if d < self.del_nodes_range:
                        node = [self.nodes[j][2], self.nodes[j][3]]
                        p_tmp = q9c.rotate([node[0], node[1], 0])
                        self.deleted_nodes.append([p_tmp[0] + trans[0], p_tmp[1] + trans[1]])
                        self.nodes.remove(self.nodes[j])
                        self.check_goal()
                        break

            if i >= 1 and abs(laser[i - 1] - laser[i]) > 1.5 and laser[i - 1] < 8.5 and laser[i] < 8.5:
                dist = (laser[i - 1] + laser[i]) / 2.0
                angl = i / (2 * 2) - 90.0
                qx, qy = calcqxqy(dist, angl, ang)

                f = True
                j = 0
                while j < len(self.nodes):
                    d = math.sqrt(
                        (self.nodes[j][2] - qx - odomX) ** 2 + (self.nodes[j][3] - qy - odomY) ** 2)
                    if d < self.node_vicinity:
                        f = False
                        break
                    j += 1
                j = 0
                while f and j < len(self.closed_nodes_rotated):
                    d = math.sqrt(
                        (self.closed_nodes_rotated[j][0] - qx - odomX) ** 2 + (self.closed_nodes_rotated[j][1] - qy - odomY) ** 2)
                    if d < self.node_vicinity:
                        f = False
                        break
                    j += 1
                j = 0
                while f and j < len(self.deleted_nodes_rotated):
                    d = math.sqrt(
                        (self.deleted_nodes_rotated[j][0] - qx - odomX) ** 2 + (self.deleted_nodes_rotated[j][1] - qy - odomY) ** 2)
                    if d < self.del_node_vicinity:
                        f = False
                        break
                    j += 1

                if f:
                    node = [qx + odomX, qy + odomY]
                    local_x = node[0]
                    local_y = node[1]
                    p_tmp = q9c.rotate([local_x, local_y, 0])
                    self.nodes.append(
                        [p_tmp[0] + trans[0], p_tmp[1] + trans[1], p_tmp[0] + trans[0], p_tmp[1] + trans[1], 0])

    def free_space_nodes(self, laser, odomX, odomY, ang, trans, q9c):
        count5 = 0
        min_d = 100.0

        for i in range(1, len(laser)):
            go = False

            if 4.5 < laser[i] < 9.9:
                count5 += 1
                min_d = min(min_d, laser[i])
                continue

            if count5 > 35:
                go = True
            else:
                count5 = 0

            if go:
                dist = 4.0
                angl = (i - count5 / 2.0) / (2 * 2) - 90.0
                count5 = 0
                min_d = 100.0
                qx, qy = calcqxqy(dist, angl, ang)

                f = True
                j = 0
                while j < len(self.nodes):
                    d = math.sqrt(
                        (self.nodes[j][2] - qx - odomX) ** 2 + (self.nodes[j][3] - qy - odomY) ** 2)
                    if d < self.node_vicinity:
                        f = False
                        break
                    j += 1
                j = 0
                while f and j < len(self.closed_nodes_rotated):
                    d = math.sqrt(
                        (self.closed_nodes_rotated[j][0] - qx - odomX) ** 2 + (self.closed_nodes_rotated[j][1] - qy - odomY) ** 2)
                    if d < self.node_vicinity:
                        f = False
                        break
                    j += 1
                j = 0
                while f and j < len(self.deleted_nodes_rotated):
                    d = math.sqrt(
                        (self.deleted_nodes_rotated[j][0] - qx - odomX) ** 2 + (self.deleted_nodes_rotated[j][1] - qy - odomY) ** 2)
                    if d < self.del_node_vicinity:
                        f = False
                        break
                    j += 1

                if f:
                    node = [qx + odomX, qy + odomY]
                    p_tmp = q9c.rotate([node[0], node[1], 0])
                    self.nodes.append(
                        [p_tmp[0] + trans[0], p_tmp[1] + trans[1], p_tmp[0] + trans[0], p_tmp[1] + trans[1], 0])

    def infinite_nodes(self, laser, odomX, odomY, ang, trans, q9c):
        tmp_i = 0
        save_i = 0

        for i in range(1, len(laser)):
            go = False

            if laser[i] < 6.9:
                if i - tmp_i > 50 and tmp_i > 0:
                    go = True
                    save_i = tmp_i
                tmp_i = i

            if go:
                dist = min(laser[save_i], laser[i])
                angl = (i - (i - save_i) / 2.0) / (2 * 2) - 90.0
                qx, qy = calcqxqy(dist, angl, ang)

                f = True
                j = 0
                while j < len(self.nodes):
                    d = math.sqrt(
                        (self.nodes[j][2] - qx - odomX) ** 2 + (self.nodes[j][3] - qy - odomY) ** 2)
                    if d < self.node_vicinity:
                        f = False
                        break
                    j += 1
                j = 0
                while f and j < len(self.closed_nodes_rotated):
                    d = math.sqrt(
                        (self.closed_nodes_rotated[j][0] - qx - odomX) ** 2 + (self.closed_nodes_rotated[j][1] - qy - odomY) ** 2)
                    if d < self.node_vicinity:
                        f = False
                        break
                    j += 1
                j = 0
                while f and j < len(self.deleted_nodes_rotated):
                    d = math.sqrt(
                        (self.deleted_nodes_rotated[j][0] - qx - odomX) ** 2 + (self.deleted_nodes_rotated[j][1] - qy - odomY) ** 2)
                    if d < self.del_node_vicinity:
                        f = False
                        break
                    j += 1

                if f:
                    node = [qx + odomX, qy + odomY]
                    p_tmp = q9c.rotate([node[0], node[1], 0])
                    self.nodes.append(
                        [p_tmp[0] + trans[0], p_tmp[1] + trans[1], p_tmp[0] + trans[0], p_tmp[1] + trans[1], 0])

    def check_pos(self, odomX, odomY):
        for i in range(len(self.nodes)):
            d = math.sqrt((self.nodes[i][2] - odomX) ** 2 + (self.nodes[i][3] - odomY) ** 2)
            if d < 0.75:
                try:
                    self.nodes.remove(self.nodes[i])
                    self.check_goal()
                except ValueError:
                    self.get_logger().warn("Attempted to remove a node that does not exist.")
                break

    def freeze(self, X, Y):
        self.last_statesX.append(X)
        self.last_statesY.append(Y)
        if len(self.last_statesX) > (self.freeze_rate - 50) and stdev(self.last_statesX) < self.stddev_threshold and \
                stdev(self.last_statesY) < self.stddev_threshold:
            return True
        return False

    def laser_check(self, lscan, col=False, colleft=False, colright=False, minleft=7.0, minright=7.0):
        min_in = self.min_in
        side_min_in = self.side_min_in

        for i in range(0, len(lscan)):
            if i < len(lscan) / 2:
                minleft = min(minleft, lscan[i])
            else:
                minright = min(minright, lscan[i])

            if len(lscan) / 4.5 < i < len(lscan) / 3.5 and lscan[i] < side_min_in:
                colleft = True
            if len(lscan) - len(lscan) / 4.5 > i > len(lscan) - len(lscan) / 3.5 and lscan[i] < side_min_in:
                colright = True

            if len(lscan) / 7 < i < len(lscan) - len(lscan) / 7 and lscan[i] < min_in:
                col = True

        return col, colleft, colright, minleft, minright
