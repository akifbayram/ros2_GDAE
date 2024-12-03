from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, ExecuteProcess, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
import os

def generate_launch_description():
    # Define paths for each launch file
    ignition_bringup_path = os.path.join(
        FindPackageShare('turtlebot4_ignition_bringup').find('turtlebot4_ignition_bringup'),
        'launch',
        'turtlebot4_ignition.launch.py'
    )

    # Path to the custom RViz configuration file
    rviz_config_path = os.path.expanduser('~/ros2_GDAE/src/GDAE/rviz/tb4.rviz')

    # Launch actions for ignition bringup
    ignition_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ignition_bringup_path),
        launch_arguments={
            'nav2': 'true',
            'slam': 'true',
            'localization': 'false',
            'rviz': 'false'
        }.items()
    )

    # Launch RViz with the custom configuration
    rviz_launch = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config_path],
        output='screen'
    )

    # Command to undock the robot after it has loaded
    undock_command = TimerAction(
        period=10.0,  # Delay to allow the robot to initialize fully
        actions=[
            ExecuteProcess(
                cmd=['ros2', 'action', 'send_goal', '/undock', 'irobot_create_msgs/action/Undock', '{}'],
                output='screen'
            )
        ]
    )

    return LaunchDescription([
        # Set environment variable for TurtleBot setup
        SetEnvironmentVariable(name='TURTLEBOT4_SETUP_BASH', value='/etc/turtlebot4/setup.bash'),
        
        # Launch ignition bringup
        ignition_launch,
        
        # Launch RViz
        rviz_launch,
        
        # Undock command
        undock_command,
    ])
