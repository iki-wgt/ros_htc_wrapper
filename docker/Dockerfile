ARG PYTORCH="1.1.0"
ARG CUDA="10.0"
ARG CUDNN="7.5"

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ARG LSB_RELEASE=xenial
ARG ROS_DISTRO=kinetic

ENV DEBIAN_FRONTEND noninteractive
ENV WAIT_FOR_ROSCORE 1

# use bash shell
RUN rm /bin/sh && ln -s /bin/bash /bin/sh

# install required tools for mmdetection
RUN apt-get update && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 lsb-core \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

# Install mmdetection
RUN conda install cython -y && conda clean --all
RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
WORKDIR /mmdetection
RUN pip install --no-cache-dir -e .

# install ROS
RUN apt-get update && apt-get -y upgrade && apt-get -y install wget apt-utils && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $LSB_RELEASE main" > /etc/apt/sources.list.d/ros-latest.list' && \
    apt-key adv --keyserver 'hkp://keyserver.ubuntu.com:80' --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654 && \
    curl -sSL 'http://keyserver.ubuntu.com/pks/lookup?op=get&search=0xC1CF6E31E6BADE8868B172B4F42ED6FBAB17C654' | apt-key add - && \
    apt-get update && apt-get -y upgrade && \
    apt-get -y install ros-$ROS_DISTRO-ros-base && \
    apt-get -y install ros-$ROS_DISTRO-rosnode && \
    apt-get -y install python-catkin-tools

# install further python packages
RUN python -m pip install catkin_pkg && \
    python -m pip install rospkg && \
    python -m pip install empy

# create catkin_ws
RUN source /opt/ros/$ROS_DISTRO/setup.sh && \
    mkdir -p /root/catkin_ws/src/prj-iki-robotics && \
    cd /root/catkin_ws/src && \
    catkin_init_workspace && \
    cd /root/catkin_ws && \
    catkin_make && \
    echo "source /root/catkin_ws/devel/setup.bash" >> /root/.bashrc

# download HTC model
RUN curl https://s3.ap-northeast-2.amazonaws.com/open-mmlab/mmdetection/models/htc/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth --output /mmdetection/htc_dconv_c3-c5_mstrain_400_1400_x101_64x4d_fpn_20e_20190408-0e50669c.pth

# copy ROS service and messages into catkin workspace
COPY ./htc_service /root/catkin_ws/src/prj-iki-robotics/htc_service
COPY ./htc_msg /root/catkin_ws/src/prj-iki-robotics/htc_msg
COPY ./htc_service_msg /root/catkin_ws/src/prj-iki-robotics/htc_service_msg

# compile catkin workspace
RUN cd /root/catkin_ws && \
    source /opt/ros/$ROS_DISTRO/setup.sh && \
    catkin_make && \
    source devel/setup.bash
