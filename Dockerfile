FROM nvidia/cuda:11.3.0-cudnn8-devel-ubuntu18.04
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget ca-certificates

RUN conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
RUN mkdir -p /root/brainstorm
COPY . /root/brainstorm
RUN chown -R $USER:$USER /root/brainstorm
WORKDIR /root/brainstorm
RUN pip install -r requirements.txt
RUN pip install -e .

RUN mkdir build
WORKDIR /home/brainstorm/build
RUN cmake ..