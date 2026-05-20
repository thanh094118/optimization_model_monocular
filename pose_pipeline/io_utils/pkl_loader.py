from pathlib import Path
from typing import Union
import joblib


def load_pkl_data(file_path):
    """Load WHAM/OpenCap PKL and return the first person payload."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError("PKL file not found: {}".format(file_path))

    data = joblib.load(file_path)
    person_data = None

    if isinstance(data, dict) or "defaultdict" in str(type(data)):
        if 0 in data:
            person_data = data[0]
        elif "0" in data:
            person_data = data["0"]
        else:
            person_data = next((v for v in data.values() if isinstance(v, dict)), None)
    elif isinstance(data, list):
        person_data = next((v for v in data if isinstance(v, dict)), None)

    if person_data is None:
        raise ValueError("Cannot read person payload from {}".format(file_path))

    for key in ("pose", "trans", "betas"):
        if key not in person_data:
            raise KeyError("Missing key {!r} in {}. Available keys: {}".format(key, file_path, list(person_data.keys())))

    return person_data
