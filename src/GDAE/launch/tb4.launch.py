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

    nav2_bringup_path = os.path.join(
        FindPackageShare('turtlebot4_navigation').find('turtlebot4_navigation'),
        'launch',
        'nav2.launch.py'
    )

    # Path to the custom RViz configuration file
    rviz_config_path = os.path.expanduser('~/ros2_GDAE/src/GDAE/rviz/tb4.rviz')

    # Path to the navigation parameter file
    nav2_params_path = os.path.expanduser('~/ros2_GDAE/src/GDAE/config/nav2.yaml')

    # Launch actions for ignition bringup
    ignition_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ignition_bringup_path),
        launch_arguments={
            'model': 'standard',
            'nav2': 'false',
            'slam': 'true',
            'localization': 'false',
            'rviz': 'false',
            'world': 'warehouse' # options: depot, maze, warehouse
        }.items()
    )

    # Launch RViz with the custom configuration
    rviz_launch = ExecuteProcess(
        cmd=['rviz2', '-d', rviz_config_path],
        output='screen'
    )

    # Launch nav2 with specified parameters
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(nav2_bringup_path),
        launch_arguments={
            'params': nav2_params_path
        }.items()
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

        # Launch nav2
        nav2_launch,
        
        # Undock command
        undock_command,
    ])
