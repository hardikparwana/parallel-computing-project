#FROM nvidia/cuda:11.6.1-cudnn8-devel-ubuntu20.04
#ARG DEBIAN_FRONTEND=noninteractive
#RUN export TZ='America/New_York'
#RUN apt-get update && apt-get install -y --no-install-recommends vim tmux gedit
FROM hardikparwana/cuda116desktop:v1
RUN apt install -y python3-pip
RUN apt install -y vim tmux

WORKDIR "/home/"
