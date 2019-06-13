FROM python:3.6.5-slim-stretch AS build-env
WORKDIR /app
LABEL maintainer="petar.gjeorgiev@interworks.com.mk"

COPY . .

RUN apt-get update && apt-get -y install libglib2.0; apt-get clean
RUN apt-get update && apt-get -y install git; apt-get clean
RUN apt-get update && apt-get -y install cmake; apt-get clean
RUN apt-get update && apt-get -y install make; apt-get clean
RUN python -m pip install --upgrade pip
RUN python -m pip install tensorflow
RUN pip install matplotlib
RUN pip install keras
RUN pip install opencv-python-headless
RUN pip install numpy
RUN pip install Pillow
RUN pip install boto3

RUN git clone https://github.com/opencv/opencv.git /usr/local/src/opencv
RUN cd /usr/local/src/opencv/ && mkdir build
RUN cd /usr/local/src/opencv/build && cmake -D CMAKE_INSTALL_TYPE=Release -D CMAKE_INSTALL_PREFIX=/usr/local/ .. && make -j4 && make install

ENTRYPOINT [ "python", "recognize.py" ]