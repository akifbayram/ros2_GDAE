from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, SetEnvironmentVariable, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution
import os

def generate_launch_description():
    # Define paths for each launch file
    ignition_bringup_path = os.path.join(
        FindPackageShare('turtlebot4_ignition_bringup').find('turtlebot4_ignition_bringup'),
        'launch',
        'turtlebot4_ignition.launch.py'
    )
    
    spawn_path = os.path.join(
        FindPackageShare('turtlebot4_ignition_bringup').find('turtlebot4_ignition_bringup'),
        'launch',
        'turtlebot4_spawn.launch.py'
    )

    # Launch actions for robot1 with maze world
    ignition_robot1 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(ignition_bringup_path),
        launch_arguments={
            # 'namespace': 'robot1',
            # 'world': 'maze' ,
            'nav2': 'true',
            'slam': 'true',
            'localization': 'false',
            'rviz': 'true'
        }.items()
    )


    return LaunchDescription([
        # Set environment variable for TurtleBot setup
        SetEnvironmentVariable(name='TURTLEBOT4_SETUP_BASH', value='/etc/turtlebot4/setup.bash'),
        
        # Launch robot1 ignition bringup with the maze world
        ignition_robot1,
    ])
