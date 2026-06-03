import pickle
import numpy as np

for path in [
    "iPhone17,1/Deployed/cameraIntrinsics.pickle",
    "iPhone17,1/Deployed_720_60fps/cameraIntrinsics.pickle"
]:
    with open(path, "rb") as f:
        d = pickle.load(f)

    print("\n", path)
    print("imageSize =", d["imageSize"])
    print("K =")
    print(d["intrinsicMat"])
    print("dist =")
    print(d["distortion"])