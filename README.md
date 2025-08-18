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

Read about the motion controllers used with the driver: 
[Controllers](https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_controllers/doc/index.html)

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


## Additional Notes


### Use the Docker Container with GUI:

Allow local connections to your X server:
```
xhost +local:docker
```
After this you can either work with the devcontainer or start a seperate docker container:

Run the container with X11 socket and environment variable:
```
docker run -it \
  --env DISPLAY=$DISPLAY \
  --env QT_X11_NO_MITSHM=1 \
  --volume /tmp/.X11-unix:/tmp/.X11-unix \
  <docker_image_name>
```
Inside the container, run rviz to check the GUI is working:
```
rviz2
```

Also check ompl is working:
```
python3 /home/ubuntu/colcon_ws/src/execute_trajectories/execute_trajectories/scripts/ompl_example.py
```


#### Run 
To launch the sim do 
```
# with default controller
ros2 launch ur_simulation_gz ur_sim_control.launch.py
# with direct joint control
ros2 launch ur_simulation_gz ur_sim_control.launch.py initial_joint_controller:="forward_position_controller"
```
To run it with direct joint control run 

```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.2.3 initial_joint_controller:=forward_position_controller
```

In another terminal do 

```
python3 ur_trajectory_with_data_recording.py 
```

or for direct joint control:

```
python3 cubic_spline.py 
```