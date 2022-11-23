if [[ -z $PATH_TO_MODEL ]]; then
  export PATH_TO_MODEL="my_model.pkl"
fi

if [[ -z $PATH_TO_TRANSFORMER ]]; then
  export PATH_TO_MODEL="my_transformer.pkl"
fi

if [ ! -f "$PATH_TO_MODEL" ]; then
    gdown 1xb9l9J_xm2AjXq9lpchfy_4yO2l45Jvn --output="$PATH_TO_MODEL"
else
    echo "My Model already downloaded"
fi

if [ ! -f "$PATH_TO_TRANSFORMER" ]; then
    gdown 1KUVUJJCZN1Bb_5oQQNATgueb9M1Ownrw --output="$PATH_TO_TRANSFORMER"
else
    echo "My Transformer already downloaded"
fi

uvicorn main:app --reload --host 0.0.0.0 --port 8000