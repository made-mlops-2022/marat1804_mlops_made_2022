# Airflow DAGS
### To start Airflow
```
export LOCAL_DIR=$(pwd)/data
export PASSWORD=special_password_for_gmail
export FERNET_KEY=$(python3 -c "from cryptography.fernet import Fernet; FERNET_KEY = Fernet.generate_key().decode(); print(FERNET_KEY)")
docker-compose up --build
```

Airflow is available at http://localhost:8080, login = admin, password = admin.


To choose model in predict_pipeline you should go to Admin -> Variables 
add the variable with name model and value that is the date of any train DAG in YYYY-MM-DD format.


### To run simple tests on dags load and structure
```
docker exec -it airflow_ml_dags_scheduler-1 bash
pip3 install pytest
python3 -m pytest --disable-warnings tests
```

