python src/train.py hydra.job.chdir=False

python src/predict.py --model_path <model_path> --transformer_path <transformer_path> --data_path <data_path> --output_path <output_path>
