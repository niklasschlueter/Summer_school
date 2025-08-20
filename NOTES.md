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
Inside the container, run:
```
rviz2
```


### Connect to the real robot: 
## If its for the very first time, follow 
https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_client_library/doc/setup.html

## Afterwards
Start up the teaching tablet, activate the robot in the bottom left corner. Then click remote control in the upper right corner.
```

```
Make sure you are connected
```
ping 192.168.2.3
```

Launch the driver
```
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=192.168.2.3
```

In the Program tab, start the "control by niklas" program (select it and press the play button).
# TODO: Fix this - should be possible from pc.


To check active controllers, type 
```
ros2 control list_controllers
```
As specified in https://docs.universal-robots.com/Universal_Robots_ROS2_Documentation/doc/ur_robot_driver/ur_robot_driver/doc/usage/move.html , we can test the connection to the robot for example as follows:

Run a test program to ensure the robot is correctly set up:
```
ros2 launch ur_robot_driver test_scaled_joint_trajectory_controller.launch.py
```

To run it iteractively with moveit, type 
```
ros2 launch ur_moveit_config ur_moveit.launch.py ur_type:=ur5e launch_rviz:=true
```
(press "plan & execute" in the lower left corner)


### Direct Joint Control
```
#ros2 control load_controller forward_position_controller
#ros2 control switch_controllers --activate forward_position_controller --deactivate scaled_joint_trajectory_controller
```

Open /opt/ros/jazzy/share/ur_simulation_gz/launch/ur_sim_control.launch.py 
- change "inital_joint_argument" from "scaled_joint_trajectory_controller" to "forward_position_controller"

OR: 

Try to launch sim as 
```
ros2 launch your_package_name launch.py initial_joint_controller:=forward_position_controller
```
(this didnt work for me for some reason)


Create a urdf file for the ur5e:
```
ros2 run xacro xacro   $(ros2 pkg prefix ur_description)/share/ur_description/urdf/ur.urdf.xacro   ur_type:=ur5e   name:=ur5e   safety_limits:=true   safety_pos_margin:=0.15   safety_k_position:=20   > ur5e.urdf
```

Make a csv file from a rosbag: 
```
python3 rosbag2csv.py bags/my_experiment
```