# Isaac Sim 기본 이미지
FROM nvcr.io/nvidia/isaac-sim:4.1.0

# ROS 2 Humble 설치를 위한 설정
ENV DEBIAN_FRONTEND=noninteractive

# 필수 패키지 설치
RUN apt-get update && apt-get install -y \
    locales \
    curl \
    gnupg2 \
    lsb-release \
    software-properties-common \
    cmake git vim python3-pip

# 로케일 설정
RUN locale-gen en_US en_US.UTF-8 && \
    update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
ENV LANG en_US.UTF-8

# ROS 2 Humble 설치
RUN curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    echo "deb http://packages.ros.org/ros2/ubuntu $(lsb_release -cs) main" > /etc/apt/sources.list.d/ros2-latest.list && \
    apt-get update && apt-get install -y \
    ros-humble-desktop \
    python3-argcomplete

# PyTorch 설치
RUN pip install torch torchvision torchaudio

# 추가적으로 필요할 수 있는 종속 패키지 설치
RUN apt-get install -y \
    python3-pip \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-vcstool

RUN ln -s /usr/bin/python3 /usr/bin/python

# ROS 2 환경 설정
RUN echo "source /opt/ros/humble/setup.bash" >> ~/.bashrc

# ROS 2 초기화
RUN rosdep init && rosdep update

# Isaaclab install

# torchrl install

# BTMS pull

# 시작 명령어 설정 (기본적으로 bash를 실행)
CMD ["bash"]
