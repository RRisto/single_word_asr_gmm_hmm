FROM ubuntu

RUN mkdir -p /opt/single_word_asr_gmm_hmm

RUN apt-get update
RUN apt-get install -y python3-pip
WORKDIR /opt/single_word_asr_gmm_hmm

RUN pip3 install --upgrade pip

RUN apt-get install -y libblas-dev liblapack-dev libatlas-base-dev gfortran

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/Helsinki
RUN apt-get -y install tzdata
RUN apt install -y libasound-dev portaudio19-dev libportaudiocpp0
RUN pip3 install --upgrade cython
RUN pip3 install -U scikit-learn

COPY requirements.txt /opt/requirements.txt
RUN pip3 install -r /opt/requirements.txt
