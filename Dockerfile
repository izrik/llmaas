#FROM python:3.11.8-alpine3.19
FROM python:3.11.8-bookworm

#RUN apk add gcc python3-dev musl-dev linux-headers make cmake
#RUN apt install

RUN pip install --upgrade pip setuptools \
    pip install wheel \
    pip install gunicorn==21.2.0

WORKDIR /llmaas

COPY requirements.txt main.py /llmaas/

RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
RUN pip install -r requirements.txt

USER 1000
CMD python main.py
