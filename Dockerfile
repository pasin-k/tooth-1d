FROM tensorflow/tensorflow:1.13.1-gpu-py3

RUN apt-get update && apt-get install -y git

RUN pip install h5py scikit-optimize

RUN git clone https://github.com/jobpasin/tooth-2d

WORKDIR .

CMD python3 train.py tooth_workspace.config