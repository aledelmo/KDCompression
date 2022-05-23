FROM pytorch/pytorch:latest

RUN mkdir -p /KDCompression
COPY requirements.txt /KDCompression

WORKDIR /KDCompression
RUN conda config --add channels conda-forge
RUN conda config --add channels pytorch
RUN conda create --name KDCompression --file requirements.txt
RUN conda activate KDCompression

EXPOSE 8888
EXPOSE 6006

CMD ["bash", "-c", "source /etc/bash.bashrc && jupyter notebook --debug --notebook-dir=/tf --ip 0.0.0.0 --allow-root"]


# docker build --build-arg https_proxy=https://10.143.11.22:3128 -f Dockerfile --rm --tag kd_compression-gpu-jupyter:latest .

# docker run -u $(id -u) --gpus all -it -v /home/imag2/IMAG2_DL/KDCompression:/KDCompression --env HTTPS_PROXY=https://10.143.11.22:3128
# -p 8888:8888 -p 6006:6006 --rm --name torch kd_compression-gpu-jupyter:latest