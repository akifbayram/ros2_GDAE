import csv
import rclpy

class GoalTimer:
    def __init__(self, node, csv_filename='gdae_goal_times.csv'):
        self.start_time = None
        self.node = node  # Reference to the ROS node
        self.csv_filename = csv_filename

    def start(self):
        self.start_time = self.node.get_clock().now()

    def stop(self, success=True):
        if self.start_time is not None:
            end_time = self.node.get_clock().now()
            elapsed_time = (end_time - self.start_time).nanoseconds / 1e9  # Convert nanoseconds to seconds
            with open(self.csv_filename, 'a', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                csvwriter.writerow([elapsed_time, 'Success' if success else 'Failure'])
            self.start_time = None
            return elapsed_time
        else:
            return None
