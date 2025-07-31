# Starting point for the summer school course
This repo is meant to be an easy way to install the UR ROS2 driver and all of its dependencies, without conflicting with anything on your personal computers. 
A simple script which executes a motion on a robot, while it is logging data from various topics is also available, but is only meant as a starting point or inspiration.

# Official UR ROS2 Driver documentation
The official documentation for the UR ROS2 driver can be found here:
[Official docs](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/index.html)

How to setup the robot:
[Setting up the robot](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/installation/robot_setup.html)

How to start the driver:
[Starting the driver](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/usage/startup.html)

Once the driver is started, you can run your own scripts in a different terminal.

# Details about the container
ROS2 will be sourced every time a new terminal is opened. If this creates issues, the line 

    source /opt/ros/jazzy/setup.bash

can be removed from the `/home/robot/.bashrc` file.

# Starting the container
The devcontainer can be started in two ways:

## Using VS Code
If using VS Code as your IDE, this devcontainer can be started by opening the folder containing this README file in VS Code, pressing F1, and pressing the option `Dev Containers: Reopen in Container`.
The container should open, and you should be good to go.

## Using the devcontainers CLI tool
Documentation for CLI tool [here](https://github.com/devcontainers/cli/blob/main/README.md).

If you are using neither, I would recommend using VS Code.