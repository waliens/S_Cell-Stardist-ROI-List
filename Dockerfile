FROM nvidia/cuda:11.6.0-runtime-ubuntu20.04

# Create the directories
RUN mkdir -p app/ models/ models/2D_versatile_HE/

RUN rm /etc/apt/sources.list.d/cuda.list
#RUN rm /etc/apt/sources.list.d/nvidia-ml.list

# Install Python3
RUN apt-get update
RUN apt-get install -y python3 python3-pip git
RUN python3 -m pip install --upgrade pip

# Install Cytomine python client
RUN pip3 install git+https://github.com/cytomine-uliege/Cytomine-python-client.git@v2.8.3

# Install Stardist and tensorflow and its dependencies
COPY requirements.txt /tmp/
RUN pip3 install -r /tmp/requirements.txt

COPY config.json /models/2D_versatile_HE/config.json
COPY thresholds.json /models/2D_versatile_HE/thresholds.json
COPY weights_best.h5 /models/2D_versatile_HE/weights_best.h5

RUN chmod 444 /models/2D_versatile_HE/config.json
RUN chmod 444 /models/2D_versatile_HE/thresholds.json
RUN chmod 444 /models/2D_versatile_HE/weights_best.h5

# Install scripts
COPY descriptor.json /app/descriptor.json
COPY run.py /app/run.py

ENTRYPOINT ["python3", "/app/run.py"]
