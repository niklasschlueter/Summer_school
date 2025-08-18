FROM ros:jazzy

#ENV DEBIAN_FRONTEND=noninteractive
#ENV PIP_BREAK_SYSTEM_PACKAGES=1

# Copy local modified install script into the image
COPY install-ompl-ubuntu.sh /home/ubuntu/install-ompl-ubuntu.sh

# Make it executable and run it
# RUN cd /home/ubuntu && chmod u+x install-ompl-ubuntu.sh && export PIP_BREAK_SYSTEM_PACKAGES=1 && ./install-ompl-ubuntu.sh --python
RUN cd /home/ubuntu && chmod u+x install-ompl-ubuntu.sh && ./install-ompl-ubuntu.sh --python

# Finding the right gazebo version: https://gazebosim.org/docs/harmonic/ros_installation/#summary-of-compatible-ros-and-gazebo-combinations

# Install build tools
RUN apt-get update && apt-get install -y \ 
  python3-colcon-common-extensions \
  python3-vcstool \
  clang-format \
  curl \
  pre-commit \
  nano \
  xvfb \
  x11vnc \
  netcat-traditional \
  ssh \
  apt-utils \
  net-tools \
  ros-jazzy-ros2-control \
  ros-jazzy-ros2-controllers \
  ros-jazzy-plotjuggler-ros \
  gedit \
  python3-pip \
  python3-venv \
  ros-jazzy-ur \
  python3-pandas \
  ros-jazzy-ros-gz \
  ros-jazzy-ur-client-library \
  ros-jazzy-ur \
  ros-jazzy-ur-simulation-gz \
  ros-jazzy-ur-controllers \
  ros-jazzy-ur-robot-driver \
  #libompl-dev \
  #ompl-demos \
  vim \
  wget \
  ros-jazzy-moveit \
  ros-jazzy-pinocchio

