FROM continuumio/anaconda3

RUN conda install pytorch-cpu torchvision-cpu -c pytorch

WORKDIR /workspace
RUN git clone https://github.com/jfemiani/srp-boxes

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
