# GDAE - Goal Driven Autonomous Exploration

> **⚠️ Work in Progress**  
> This setup is currently under active development and may not be fully functional yet.

---

## Acknowledgments

 - [**reiniscimurs/GDAE**](https://github.com/reiniscimurs/GDAE)
 
---

## Requirements

- **ROS2 Humble**
- **TurtleBot Packages**
  - `turtlebot4` (for TurtleBot 4)
  - `turtlebot3` (for TurtleBot 3)
---

## Installation

1. **Clone the Repository:**

    ```bash
    git clone https://github.com/akifbayram/ros2_GDAE.git
    cd ros2_GDAE
    ```

2. **Build and Source Workspace:**

    ```bash
    colcon build
    source install/setup.bash
    ```

---

## Usage

### 1. **Launch the Simulation and Environment Nodes**

Start the TurtleBot simulation along with necessary nodes for SLAM and navigation:

```bash
source /opt/ros/humble/setup.bash
cd ~/ros2_GDAE
source install/setup.bash
ros2 launch gdae tb4.launch.py
```

### 2. **Start the Autonomous Exploration**

Run the Goal Driven Autonomous Exploration node:

```bash
ros2 run gdae GDAM.py
```

**Arguments:**
- `--x`: X-coordinate of the global goal.
- `--y`: Y-coordinate of the global goal.

**Example:**

```bash
ros2 run gdae GDAM.py --x 5.0 --y 0.0
```