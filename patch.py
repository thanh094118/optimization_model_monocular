import re

cam_ids = ["cam1", "cam2"]
for cam_id in cam_ids:
    m = re.search(r'\d+', cam_id)
    print(m.group())
