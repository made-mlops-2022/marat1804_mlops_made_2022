FROM python:3.9.15-slim-bullseye
RUN python3 -m pip install --upgrade pip

COPY train_test_split.py requirements.txt /

RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "train_test_split.py"]