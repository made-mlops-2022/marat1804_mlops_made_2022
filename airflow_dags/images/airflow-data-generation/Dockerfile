FROM python:3.9.15-slim-bullseye
RUN python3 -m pip install --upgrade pip

COPY data_generation.py input_data.csv requirements.txt /

RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "data_generation.py"]