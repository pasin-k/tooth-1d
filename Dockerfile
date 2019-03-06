FROM tensorflow/tensorflow:1.12.0-gpu-py3

RUN apt-get update && apt-get install -y git

RUN pip install h5py scikit-optimize

RUN git clone https://github.com/jobpasin/tooth-2d

WORKDIR tooth-2d

CMD python3 train.py tooth_workspace.config