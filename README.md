# GDAE - Goal Driven Autonomous Exploration

## Acknowledgments

 - [**reiniscimurs/GDAE**](https://github.com/reiniscimurs/GDAE) - ROS Melodic
 - [**reiniscimurs/DRL-Robot-Navigation-ROS2**](https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2) – Used to train the navigation model

---

## Overview

GDAE is a goal-driven autonomous exploration and mapping system designed for unknown environments. The system integrates reactive and planned navigation to achieve autonomous exploration without prior knowledge of the map. A deep reinforcement learning (DRL) policy serves as the local navigation layer, guiding the robot towards intermediate goals while avoiding obstacles. The global navigation layer mitigates local optima by strategically selecting global goals. 

This repository is an **ongoing attempt** to port the original ROS Melodic implementation to ROS2 Humble. The original code relied on `move_base`, whereas this port uses Nav2 for global navigation and integrates a PyTorch-based DRL model for local control.

**Model Training Note**: The DRL model included in this repository was trained using the DRL-Robot-Navigation-ROS2 project. For demonstration and testing purposes, this model has been trained for a minimal number of epochs. Consequently, it is not indicative of the performance of a fully-trained policy. Instead, it serves as a placeholder to illustrate the integration and functionality of the system within the ROS2 framework.

### **Demonstration without navigation policy**

<img src="media/gdae.gif" width="640" />

---

## Requirements

- **ROS2 Humble**
- `turtlebot4` (for TurtleBot 4 hardware or simulation)
- Python 3.8+
- PyTorch >= 1.7.0

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/akifbayram/ros2_GDAE.git
cd ros2_GDAE
```

### 2. Build and Source Workspace

```bash
colcon build
source install/setup.bash
```

---

## Usage

### 1. Launch the Simulation and Environment Nodes

Start the TurtleBot simulation along with necessary nodes for SLAM, Nav2, and Rviz2:

```bash
cd ~/ros2_GDAE &&
source install/setup.bash &&
source /etc/turtlebot4/setup.bash &&
ros2 launch gdae tb4.launch.py 
```

The TurtleBot4 will attempt to undock as part of the launch script. If the TurtleBot4 fails to undock, it can be manually acheived using the Gazebo Ignition interface (click 4 then two dots) or the command below:

```bash
ros2 action send_goal /undock irobot_create_msgs/action/Undock "{}"
```

---

### 2. Start the Autonomous Exploration

In another terminal, run the Goal Driven Autonomous Exploration node:

```bash
cd ~/ros2_GDAE &&
source install/setup.bash &&
source /etc/turtlebot4/setup.bash &&
ros2 run gdae GDAM
```

**Arguments:**
- `--x`: X-coordinate of the global goal.
- `--y`: Y-coordinate of the global goal.

**Example:**

```bash
ros2 run gdae GDAM --x 5.0 --y 0.0
```

---

## Differences from the Original ROS Melodic Implementation

- **Navigation Stack:**  
  The original ROS1 system relied on `move_base` and `actionlib`. In ROS2, Nav2’s `navigate_to_pose` action server is used for global navigation commands.
  
- **DRL Model Integration:**  
  Originally, TensorFlow/TFLearn was used. Now, PyTorch is employed, and the model provided was trained using the [DRL-Robot-Navigation-ROS2](https://github.com/reiniscimurs/DRL-Robot-Navigation-ROS2) repository.

---

## References

- IEEE Robotics and Automation Letters, ICRA 2022  
   _Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning_  
   **Authors**: Reinis Cimurs, Il Hong Suh, Jin Han Lee  
   DOI: [10.1109/LRA.2021.3133591](https://doi.org/10.1109/LRA.2021.3133591)

```bibtex
@ARTICLE{9645287,
  author={Cimurs, Reinis and Suh, Il Hong and Lee, Jin Han},
  journal={IEEE Robotics and Automation Letters}, 
  title={Goal-Driven Autonomous Exploration Through Deep Reinforcement Learning}, 
  year={2022},
  volume={7},
  number={2},
  pages={730-737},
  doi={10.1109/LRA.2021.3133591}}
```

For additional videos and experiments, visit the [original repository](https://github.com/reiniscimurs/GDAE).

---