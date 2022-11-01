# Simple ML Project with tests
### 1. Create and activate virtual environment
```
python3 -m venv venv
source venv/bin/activate
```

### 2. Install requirements
```
pip install -r requirements.txt
```

### 3. Download data
Get the data:
```
dvc pull -r gdrive
```

### 4. Run training pipeline
Working directory should be `ml_project` (on default model is `logistic_regression`)
```
python src/train.py hydra.job.chdir=False
```
To train k_neighbours_classifier run:

```
python src/train.py model=k_neighbours_classifier hydra.job.chdir=False
```
### 5. Run prediction pipeline
Working directory should be `ml_project`
```
python src/predict.py --model_path <model_path> --transformer_path <transformer_path> --data_path <data_path> --output_path <output_path>
```
To get more info about the options use:
```
python src/predict.py --help
```
To run default example:
```
python src/predict.py
```

### 6. Generate synthetic data for tests
Generate synthetic data:
```
python tests/generate_data.py
```
To check some statistics relative to origin dataset check `tests/data/statistics`

And the data will be in `tests/data/synthetic_data.csv`

### 7. Run tests

Run tests (may take a while):
```
python -m unittest tests
```



    