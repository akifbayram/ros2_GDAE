import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import QoSProfile, qos_profile_sensor_data

import numpy as np
import math
from statistics import stdev
from collections import deque

from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from nav2_msgs.action import NavigateToPose

import tf_transformations
import tf2_ros
from tf2_ros import TransformException
from pyquaternion import Quaternion
from script.timer import GoalTimer

def rotate(point, angle):
    """Rotate a point by a given angle."""
    x, y = point
    cos_ang = math.cos(angle)
    sin_ang = math.sin(angle)
    xx = x * cos_ang - y * sin_ang
    yy = x * sin_ang + y * cos_ang
    return xx, yy

def calcqxqy(dist, angl, ang):
    """
    Calculate rotated coordinates based on distance and angles.
    dist: distance measurement from laser scan
    angl: angle from laser scan segment
    ang: robot's current orientation angle
    """
    angl = math.radians(angl)
    angle = angl + ang
    # Normalize angle to [-pi, pi]
    angle = normalize_angle(angle)
    qx, qy = rotate([dist, 0], angle)
    return qx, qy

def normalize_angle(angle):
    """Normalize an angle to the range [-pi, pi]."""
    return (angle + np.pi) % (2 * np.pi) - np.pi

class ImplementEnv(Node):
    def __init__(self, args):

        super().__init__('gdae')
        self.get_logger().info('ImplementEnv node has been initialized.')

        # Initialize GoalTimer
        self.goal_timer = GoalTimer(self)

        # Initialize counters
        self.countZero = 0
        self.count_turn = -100

        # Initialize parameters from args
        self.node_vicinity = args.node_vicinity
        self.del_node_vicinity = args.deleted_node_vicinity
        self.min_in = args.min_in
        self.side_min_in = args.side_min_in
        self.del_nodes_range = args.delete_nodes_range
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

        self.angle = 0.0
        self.odomX = 0.0
        self.odomY = 0.0
        self.linearLast = 0.0
        self.angularLast = 0.0

        self.LaserData = None
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

        qos = QoSProfile(depth=10)
        sensor_qos = qos_profile_sensor_data

        # Publishers
        self.vel_pub = self.create_publisher(Twist, '/cmd_vel', qos)

        topic_nodes_pub = 'vis_mark_array_node'
        self.nodes_pub = self.create_publisher(Marker, topic_nodes_pub, qos)

        topic_nodes_pub_closed = 'vis_mark_array_node_closed'
        self.nodes_pub_closed = self.create_publisher(Marker, topic_nodes_pub_closed, qos)

        topic_map_nodes = 'vis_mark_array_map_nodes'
        self.map_nodes_viz = self.create_publisher(Marker, topic_map_nodes, qos)

        topic = 'vis_mark_array'
        self.publisher = self.create_publisher(MarkerArray, topic, qos)

        self.global_goal_publisher = self.create_publisher(MarkerArray, 'global_goal_publisher', qos)

        # Subscriptions
        self.navLaser = self.create_subscription(LaserScan, '/scan', self.Laser_callback, sensor_qos)

        self.navOdom = self.create_subscription(Odometry, '/odom', self.Odom_callback, qos)

        # Action Client
        self.client = ActionClient(self, NavigateToPose, 'navigate_to_pose')
        while not self.client.wait_for_server(timeout_sec=1.0):
            self.get_logger().info('Waiting for action server navigate_to_pose...')
        self.get_logger().info('Action server navigate_to_pose is now available.')

        # TF Listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Start the goal timer
        self.goal_timer.start()

    def Laser_callback(self, l):
        self.LaserData = l

    def Odom_callback(self, o):
        self.OdomData = o

    def step(self, act):
        rplidar = None
        self.map_nodes.clear()

        # Wait for laser scan
        while rplidar is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            rplidar = self.LaserData

        dataOdom = None
        # Wait for odometry data
        while dataOdom is None:
            rclpy.spin_once(self, timeout_sec=0.1)
            dataOdom = self.OdomData

        # Process single laser scan
        laser_in = np.array(rplidar.ranges)
        laser_in = np.clip(laser_in, 0.0, 10.0)    # Set max range to 10.0
        laser_in = np.nan_to_num(laser_in, nan=10.0)  # Replace NaNs with 10.0

        # Define the number of segments
        num_segments = 19
        segment_length = len(laser_in) // num_segments
        laser_state = []
        for i in range(num_segments):
            start = i * segment_length
            # Ensure the last segment includes any remaining points
            end = (i + 1) * segment_length if i < num_segments - 1 else len(laser_in)
            segment = laser_in[start:end]
            min_val = np.min(segment)
            laser_state.append(min_val / 10.0)  # Normalize

        laser_state = np.array(laser_state)

        # Laser collision check
        col, colleft, colright, minleft, minright = self.laser_check(laser_in)

        # Update odometry
        self.odomX = dataOdom.pose.pose.position.x
        self.odomY = dataOdom.pose.pose.position.y
        quaternion = (
            dataOdom.pose.pose.orientation.x,
            dataOdom.pose.pose.orientation.y,
            dataOdom.pose.pose.orientation.z,
            dataOdom.pose.pose.orientation.w)
        euler = tf_transformations.euler_from_quaternion(quaternion)
        self.angle = round(euler[2], 4)

        # Transform from map to odom
        try:
            transform = self.tf_buffer.lookup_transform('map', 'odom', rclpy.time.Time())
            trans = [transform.transform.translation.x, transform.transform.translation.y, transform.transform.translation.z]
            rot = [transform.transform.rotation.x, transform.transform.rotation.y, transform.transform.rotation.z, transform.transform.rotation.w]
        except TransformException as ex:
            self.get_logger().error(f'Could not transform odom to map: {ex}')
            return laser_state, [0, 0]

        # Update global goal based on transform
        local_x = self.original_goal_x - trans[0]
        local_y = self.original_goal_y - trans[1]
        eul = tf_transformations.euler_from_quaternion(rot)
        q8c = Quaternion(axis=[0.0, 0.0, 1.0], radians=-eul[2])
        q9c = Quaternion(axis=[0.0, 0.0, 1.0], radians=eul[2])
        p_tmp = q8c.rotate([local_x, local_y, 0])
        self.global_goal_x = p_tmp[0]
        self.global_goal_y = p_tmp[1]

        # Calculate distance to global goal
        global_goal_distance = math.hypot(self.odomX - self.global_goal_x, self.odomY - self.global_goal_y)

        if global_goal_distance < 1.5:
            # Stop the robot
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.0
            self.vel_pub.publish(vel_cmd)
            self.get_logger().info('Published stop command.')
            # Stop the goal timer
            elapsed_time = self.goal_timer.stop(success=True)
            if elapsed_time is not None:
                self.get_logger().info(f'Goal achieved in {elapsed_time} seconds.')
            else:
                self.get_logger().info('Timer was already stopped.')
            return laser_state, [0, 0]

        # Update nodes based on laser scan
        self.new_nodes(laser_in, self.odomX, self.odomY, self.angle, trans, q9c)
        self.free_space_nodes(laser_in, self.odomX, self.odomY, self.angle, trans, q9c)
        self.infinite_nodes(laser_in, self.odomX, self.odomY, self.angle, trans, q9c)

        if not self.nodes:
            vel_cmd = Twist()
            vel_cmd.linear.x = 0.0
            vel_cmd.angular.z = 0.1
            self.vel_pub.publish(vel_cmd)
            self.get_logger().info("No nodes available. Rotating to find nodes.")
            return laser_state, [0, 0]

        # Update node positions based on transform
        for i_node in range(len(self.nodes)):
            if self.nodes[i_node][4] == 0:
                node = [self.nodes[i_node][0], self.nodes[i_node][1]]
                local_x = node[0] - trans[0]
                local_y = node[1] - trans[1]
                p_tmp = q8c.rotate([local_x, local_y, 0])
                self.nodes[i_node][2] = p_tmp[0]
                self.nodes[i_node][3] = p_tmp[1]

        # Rotate closed nodes
        self.closed_nodes_rotated.clear()
        for node in self.closed_nodes:
            local_x = node[0] - trans[0]
            local_y = node[1] - trans[1]
            p_tmp = q8c.rotate([local_x, local_y, 0])
            self.closed_nodes_rotated.append([p_tmp[0], p_tmp[1]])

        # Rotate deleted nodes
        self.deleted_nodes_rotated.clear()
        for node in self.deleted_nodes:
            local_x = node[0] - trans[0]
            local_y = node[1] - trans[1]
            p_tmp = q8c.rotate([local_x, local_y, 0])
            self.deleted_nodes_rotated.append([p_tmp[0], p_tmp[1]])

        if not self.last_states:
            self.last_states = self.freeze(self.odomX, self.odomY)
            self.count_turn = 8

        self.goalX = self.nodes[self.g_node][2]
        self.goalY = self.nodes[self.g_node][3]

        Dist = math.hypot(self.odomX - self.goalX, self.odomY - self.goalY)

        skewX = self.goalX - self.odomX
        skewY = self.goalY - self.odomY

        beta = math.atan2(skewY, skewX)
        beta2 = normalize_angle(beta - self.angle)

        linear = act[0]
        angular = act[1]

        # Recover velocities based on collision
        linear, angular = self.recover(linear, angular, minleft, minright, colleft, colright, col)

        vel_cmd = Twist()
        vel_cmd.linear.x = float(linear)      # Ensure float type
        vel_cmd.angular.z = float(angular)   # Ensure float type

        if abs(angular) < 0.3:
            vel_cmd.angular.z = np.sign(angular) * (angular ** 2) / 0.3
        else:
            vel_cmd.angular.z = float(angular)  # Ensure float type

        self.vel_pub.publish(vel_cmd)

        self.linearLast = linear
        self.angularLast = angular

        # Visualization markers
        self.publish_markers()

        if Dist < 1.0:
            self.get_logger().info("Distance to goal less than 1.0. Changing goal.")
            self.change_goal()
            self.countZero = 0

        if Dist > 5.0:
            self.update_goal_from_path(trans, q8c)

        self.check_pos(self.odomX, self.odomY)

        if self.countZero % 30 == 0:
            self.add_current_position_to_closed_nodes(trans, q9c)

        if self.countZero % self.update_rate == 0:
            self.check_goal()

        if self.countZero > self.remove_rate:
            self.get_logger().info('countZero exceeded remove_rate. Changing goal.')
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
                self.lock = 1 if minright - minleft > 0 else -1
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
        gX = self.global_goal_x
        gY = self.global_goal_y
        d1 = np.hypot(candidateX - odomX, candidateY - odomY)
        d2 = np.hypot(candidateX - gX, candidateY - gY)
        if 5 < d1 < 10:
            d1 = 5
        if d1 < 5:
            d1 = 0
        d = d1 + d2
        return d

    def change_goal(self):
        """Change the current navigation goal."""
        self.get_logger().info('Changing goal.')
        if self.nodes:
            try:
                self.nodes.remove(self.nodes[self.g_node])
                self.get_logger().info(f'Removed node {self.g_node} from nodes.')
                self.check_goal()
            except IndexError:
                pass

    def check_goal(self):
        if not self.nodes:
            return

        min_d = self.heuristic(self.odomX, self.odomY, self.nodes[0][2], self.nodes[0][3])
        node_out = 0

        for i in range(len(self.nodes)):
            d = self.heuristic(self.odomX, self.odomY, self.nodes[i][2], self.nodes[i][3])
            if d < min_d:
                min_d = d
                node_out = i

        self.g_node = node_out
        self.get_logger().info(f'Setting node {self.g_node} as the new goal.')

        # Send goal to the navigation action server
        goal_msg = NavigateToPose.Goal()
        goal_msg.pose.header.frame_id = 'map'  # Ensure 'map' frame exists
        goal_msg.pose.header.stamp = self.get_clock().now().to_msg()
        goal_msg.pose.pose.position.x = self.nodes[self.g_node][2]
        goal_msg.pose.pose.position.y = self.nodes[self.g_node][3]
        goal_msg.pose.pose.orientation.w = 1.0
        self.client.send_goal_async(goal_msg)
        self.get_logger().info(self.colorize(f'Sent NavigateToPose goal: X={goal_msg.pose.pose.position.x}, Y={goal_msg.pose.pose.position.y}', 'green'))

    def colorize(self, text, color):
        """Add ANSI color codes to the text."""
        colors = {
            'red': '\033[91m',
            'green': '\033[92m',
            'yellow': '\033[93m',
            'blue': '\033[94m',
            'magenta': '\033[95m',
            'cyan': '\033[96m',
            'white': '\033[97m',
            'reset': '\033[0m'
        }
        return f"{colors.get(color, colors['reset'])}{text}{colors['reset']}"

    def new_nodes(self, laser, odomX, odomY, ang, trans, q9c):
        for i in range(1, len(laser)):
            if len(self.nodes) > 0 and laser[i] < 7.5:
                dist = laser[i]
                angl = i / 4.0 - 90.0
                qx, qy = calcqxqy(dist, angl, ang)

                self.map_nodes.append([qx + odomX, qy + odomY])

                candidate_pos = np.array([qx + odomX, qy + odomY])

                if self.is_node_in_vicinity(candidate_pos):
                    node = [qx + odomX, qy + odomY]
                    local_x = node[0]
                    local_y = node[1]
                    p_tmp = q9c.rotate([local_x, local_y, 0])
                    self.nodes.append(
                        [p_tmp[0] + trans[0], p_tmp[1] + trans[1], p_tmp[0] + trans[0], p_tmp[1] + trans[1], 0])
                    self.get_logger().info(self.colorize(f'Added new node: X={p_tmp[0] + trans[0]}, Y={p_tmp[1] + trans[1]}', 'blue'))

    def free_space_nodes(self, laser, odomX, odomY, ang, trans, q9c):
        """Add free space nodes based on laser scan data."""
        self.get_logger().debug('Adding free space nodes based on laser scan.')
        count5 = 0

        for i in range(1, len(laser)):
            go = False

            if 4.5 < laser[i] < 9.9:
                count5 += 1
                continue

            if count5 > 35:
                go = True
            else:
                count5 = 0

            if go:
                dist = 4.0
                angl = (i - count5 / 2.0) / (2 * 2) - 90.0
                count5 = 0
                qx, qy = calcqxqy(dist, angl, ang)

                candidate_pos = np.array([qx + odomX, qy + odomY])

                if self.is_node_in_vicinity(candidate_pos):
                    node = [qx + odomX, qy + odomY]
                    p_tmp = q9c.rotate([node[0], node[1], 0])
                    self.nodes.append(
                        [p_tmp[0] + trans[0], p_tmp[1] + trans[1], p_tmp[0] + trans[0], p_tmp[1] + trans[1], 0])
                    self.get_logger().info(self.colorize(
                        f'Added free space node: X={p_tmp[0] + trans[0]}, Y={p_tmp[1] + trans[1]}', 'blue'))

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

                candidate_pos = np.array([qx + odomX, qy + odomY])

                if self.is_node_in_vicinity(candidate_pos):
                    node = [qx + odomX, qy + odomY]
                    p_tmp = q9c.rotate([node[0], node[1], 0])
                    self.nodes.append(
                        [p_tmp[0] + trans[0], p_tmp[1] + trans[1], p_tmp[0] + trans[0], p_tmp[1] + trans[1], 0])
                    self.get_logger().info(self.colorize(
                        f'Added infinite node: X={p_tmp[0] + trans[0]}, Y={p_tmp[1] + trans[1]}', 'blue'))

    def is_node_in_vicinity(self, candidate_pos):
        """Check if a node is within vicinity of existing, closed, or deleted nodes."""
        if len(self.nodes) > 0:
            nodes_positions = np.array([[node[2], node[3]] for node in self.nodes])
            distances = np.linalg.norm(nodes_positions - candidate_pos, axis=1)
            if np.any(distances < self.node_vicinity):
                return False

        if len(self.closed_nodes_rotated) > 0:
            closed_positions = np.array(self.closed_nodes_rotated)
            distances = np.linalg.norm(closed_positions - candidate_pos, axis=1)
            if np.any(distances < self.node_vicinity):
                return False

        if len(self.deleted_nodes_rotated) > 0:
            deleted_positions = np.array(self.deleted_nodes_rotated)
            distances = np.linalg.norm(deleted_positions - candidate_pos, axis=1)
            if np.any(distances < self.del_node_vicinity):
                return False

        return True

    def check_pos(self, odomX, odomY):
        """Remove nodes that the robot has reached."""
        candidate_pos = np.array([odomX, odomY])
        nodes_positions = np.array([[node[2], node[3]] for node in self.nodes])
        distances = np.linalg.norm(nodes_positions - candidate_pos, axis=1)
        indices = np.where(distances < 0.75)[0]
        if indices.size > 0:
            self.nodes.remove(self.nodes[indices[0]])
            self.check_goal()

    def freeze(self, X, Y):
        """Determine if the robot's state is stable based on standard deviation thresholds."""
        self.last_statesX.append(X)
        self.last_statesY.append(Y)

        if len(self.last_statesX) == self.freeze_rate and \
                stdev(self.last_statesX) < self.stddev_threshold and \
                stdev(self.last_statesY) < self.stddev_threshold:
            return True
        return False

    def laser_check(self, lscan):
        """Check for collisions based on laser scan data."""
        col = False
        colleft = False
        colright = False
        minleft = 7.0
        minright = 7.0
        min_in = self.min_in
        side_min_in = self.side_min_in

        lscan = np.array(lscan)
        mid_index = len(lscan) // 2
        minleft = np.min(lscan[:mid_index])
        minright = np.min(lscan[mid_index:])

        colleft_indices = slice(int(len(lscan) / 4.5), int(len(lscan) / 3.5))
        colright_indices = slice(int(len(lscan) - len(lscan) / 3.5), int(len(lscan) - len(lscan) / 4.5))
        front_indices = slice(int(len(lscan) / 7), int(len(lscan) - len(lscan) / 7))

        if np.any(lscan[colleft_indices] < side_min_in):
            colleft = True

        if np.any(lscan[colright_indices] < side_min_in):
            colright = True

        if np.any(lscan[front_indices] < min_in):
            col = True

        return col, colleft, colright, minleft, minright

    def publish_markers(self):
        """Publish visualization markers for nodes, closed nodes, and goals."""
        # Nodes
        sphere_list = Marker()
        sphere_list.header.frame_id = "odom"
        sphere_list.type = Marker.SPHERE_LIST
        sphere_list.action = Marker.ADD
        sphere_list.scale.x = 0.3
        sphere_list.scale.y = 0.3
        sphere_list.scale.z = 0.3
        sphere_list.color.a = 1.0
        sphere_list.color.r = 0.0
        sphere_list.color.g = 0.0
        sphere_list.color.b = 1.0
        sphere_list.pose.orientation.w = 1.0

        for node in self.nodes:
            p = Point()
            p.x = float(node[2])  # Ensure float type
            p.y = float(node[3])  # Ensure float type
            p.z = 0.0             # Ensure float type
            sphere_list.points.append(p)
        self.nodes_pub.publish(sphere_list)

        # Closed Nodes
        closed_list = Marker()
        closed_list.header.frame_id = "odom"
        closed_list.type = Marker.SPHERE_LIST
        closed_list.action = Marker.ADD
        closed_list.scale.x = 0.2
        closed_list.scale.y = 0.2
        closed_list.scale.z = 0.2
        closed_list.color.a = 1.0
        closed_list.color.r = 1.0
        closed_list.color.g = 0.5
        closed_list.color.b = 0.5
        closed_list.pose.orientation.w = 1.0

        for node in self.closed_nodes_rotated:
            p = Point()
            p.x = float(node[0])  # Ensure float type
            p.y = float(node[1])  # Ensure float type
            p.z = 0.0             # Ensure float type
            closed_list.points.append(p)
        self.nodes_pub_closed.publish(closed_list)

        # Map Nodes
        map_list = Marker()
        map_list.header.frame_id = "odom"
        map_list.type = Marker.SPHERE_LIST
        map_list.action = Marker.ADD
        map_list.scale.x = 0.1
        map_list.scale.y = 0.1
        map_list.scale.z = 0.1
        map_list.color.a = 1.0
        map_list.color.r = 1.0
        map_list.color.g = 0.0
        map_list.color.b = 0.0
        map_list.pose.orientation.w = 1.0

        for node in self.map_nodes:
            p = Point()
            p.x = float(node[0])  # Ensure float type
            p.y = float(node[1])  # Ensure float type
            p.z = 0.0             # Ensure float type
            map_list.points.append(p)
        self.map_nodes_viz.publish(map_list)

        # Current Goal Marker
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

        # Global Goal Marker
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

    def update_goal_from_path(self, trans, q8c):
        p_data = self.PathData

        try:
            path_len = len(p_data.poses) - 1
        except AttributeError:
            path_len = 0

        c_p = 0
        while c_p <= path_len and path_len > 0:
            plan_x = p_data.poses[c_p].pose.position.x
            plan_y = p_data.poses[c_p].pose.position.y
            node = [plan_x, plan_y]
            local_x = node[0] - trans[0]
            local_y = node[1] - trans[1]
            p_tmp = q8c.rotate([local_x, local_y, 0])

            d = np.hypot(self.odomX - p_tmp[0], self.odomY - p_tmp[1])
            if d > 4.0 or (c_p == path_len and d > 1.5):
                f = True
                for j in range(len(self.nodes)):
                    check_d = np.hypot(self.nodes[j][2] - p_tmp[0], self.nodes[j][3] - p_tmp[1])
                    if check_d < 1.0:
                        self.g_node = j
                        f = False
                        break
                if f:
                    self.nodes.append([plan_x, plan_y, plan_x, plan_y, 0])
                    self.g_node = len(self.nodes) - 1
                break
            c_p += 1

    def add_current_position_to_closed_nodes(self, trans, q9c):
        f = True
        candidate_pos = np.array([self.odomX, self.odomY])
        closed_positions = np.array(self.closed_nodes_rotated)
        distances = np.linalg.norm(closed_positions - candidate_pos, axis=1)
        if np.any(distances < 0.85):
            f = False

        if f:
            node = [self.odomX, self.odomY]
            p_tmp = q9c.rotate([node[0], node[1], 0])
            self.closed_nodes.append([p_tmp[0] + trans[0], p_tmp[1] + trans[1]])
