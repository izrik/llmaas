FROM python:3.12.7-bookworm

RUN pip install --upgrade pip setuptools ; \
    pip install wheel ; \
    pip install gunicorn==21.2.0

RUN pip3 install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2

WORKDIR /llmaas
COPY requirements.txt /llmaas/
RUN pip install -r requirements.txt

RUN mkdir /llmaas/models
ENV PROJECT_ROOT_PATH="/llmaas"

COPY main.py /llmaas/

ARG VERSION=0.0
LABEL Name=llmaas
LABEL Version=$VERSION
RUN echo "__version__ = '$VERSION'" > /llmaas/__version__.py

USER 1000
CMD python main.py
