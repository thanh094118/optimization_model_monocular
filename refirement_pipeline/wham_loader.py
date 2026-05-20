import pickle
import collections

import joblib


def load_wham_output(path):
    try:
        data = joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            data = pickle.load(f)

    if isinstance(data, (dict, collections.defaultdict)):
        return [data[k] for k in sorted(data.keys())]

    return data