# GDAE - Goal Driven Autonomous Exploration

> **⚠️ Work in Progress**  
> This setup is currently under active development and may not be fully functional yet.

---

## Acknowledgments

 - [**reiniscimurs/GDAE**](https://github.com/reiniscimurs/GDAE)
 
---

## Overview

GDAE is a goal-driven autonomous exploration and mapping system designed for unknown environments. The system integrates reactive and planned navigation to achieve autonomous exploration without prior knowledge of the map. A deep reinforcement learning (DRL) policy serves as the local navigation layer, guiding the robot towards intermediate goals while avoiding obstacles. The global navigation layer mitigates local optima by strategically selecting global goals. 

This repository is an **ongoing attempt** to port the original ROS Melodic implementation to ROS2 Humble.

### **Demonstration without navigation policy**

<img src="media/gdae.gif" width="640" />

---

## Requirements

- **ROS2 Humble**
- `turtlebot4` (for TurtleBot 4 hardware or simulation)
- Python 3.8+
- TensorFlow 2.x

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

The TurtleBot4 will undock as part of the launch script.

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