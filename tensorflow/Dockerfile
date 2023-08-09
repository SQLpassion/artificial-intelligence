FROM tensorflow/tensorflow:latest-gpu-jupyter

RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y net-tools
RUN apt-get install -y iputils-ping
RUN pip install --upgrade pip
RUN pip install scipy