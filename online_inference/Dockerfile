FROM python:3.9.15-slim-bullseye
RUN python3 -m pip install --upgrade pip

COPY main.py schemas.py start.sh requirements.txt test_app.py ./app/

WORKDIR /app

RUN pip3 install -r requirements.txt
RUN pip3 install gdown

ENV PATH_TO_MODEL="my_model.pkl"
ENV PATH_TO_TRANSFORMER="my_transformer.pkl"

EXPOSE 8000

CMD ["bash", "start.sh"]