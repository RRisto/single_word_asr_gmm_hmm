FROM python3.7

RUN mkdir -p /opt/single_word_asr_gmm_hmm

RUN apt-get update
RUN apt-get install -y python3-pip
WORKDIR /opt/single_word_asr_gmm_hmm

RUN pip3 install --upgrade pip
RUN pip3 install -r requirements.txt
