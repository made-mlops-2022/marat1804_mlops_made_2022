import pickle


def load_pickle_file(path):
    with open(path, 'rb') as f:
        model = pickle.load(f)
    return model


def save_pickle_file(path, model):
    with open(path, 'wb') as f:
        pickle.dump(model, f)
