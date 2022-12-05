FROM python:3.9.15-slim-bullseye
RUN python3 -m pip install --upgrade pip

COPY validate_model.py requirements.txt /

RUN pip3 install -r requirements.txt
ENTRYPOINT ["python3", "validate_model.py"]