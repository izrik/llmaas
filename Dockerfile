#FROM python:3.11.8-alpine3.19
FROM python:3.11.8-bookworm

#RUN apk add gcc python3-dev musl-dev linux-headers make cmake
#RUN apt install

RUN pip install --upgrade pip setuptools \
    pip install wheel \
    pip install gunicorn==21.2.0
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

WORKDIR /llmaas

COPY requirements.txt /llmaas/
RUN pip install -r requirements.txt

COPY main.py /llmaas/

USER 1000
CMD python main.py
