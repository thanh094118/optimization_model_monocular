import os
import json
import numpy as np
from os.path import join


class FixParams:
    """
    Module để fix params bằng cách thay thế arm poses từ file JSON
    Chỉ thay thế 12 giá trị tại index 51-62 (left_elbow, right_elbow, left_wrist, right_wrist)
    """
    # Joint indices trong poses vector (69 phần tử)
    # left_elbow:  joint 18 → index 51, 52, 53
    # right_elbow: joint 19 → index 54, 55, 56
    # left_wrist:  joint 20 → index 57, 58, 59
    # right_wrist: joint 21 → index 60, 61, 62
    ARM_INDICES = list(range(0, 69))  # [51, 52, ..., 62]

    def __init__(self, input_dir='output/poses', output_dir='fix', **kwargs):
        """
        Args:
            input_dir:  thư mục chứa file JSON với poses cần lấy (thay cho hybrik_res_dir)
            output_dir: thư mục lưu params đã fix
        """
        self.input_dir  = input_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def fix_single_param(self, param, imgname):
        """
        Fix params cho một ảnh – chỉ thay index 51-62.

        Args:
            param:   dict chứa params từ infer module (phải có key 'poses')
            imgname: tên file ảnh (str hoặc list[str])

        Returns:
            dict chứa params đã được fix
        """
        current_params = param.copy()

        # Lấy base name (không extension)
        if isinstance(imgname, str):
            base_name = os.path.splitext(os.path.basename(imgname))[0]
        else:
            base_name = os.path.splitext(os.path.basename(imgname[0]))[0]

        src_json = join(self.input_dir, f'{base_name}.json')

        if os.path.exists(src_json):
            with open(src_json, 'r') as f:
                src_data = json.load(f)

            if 'poses' in src_data:
                src_poses = np.array(src_data['poses'])

                # Ambil poses hiện tại từ PARE
                cur_poses = current_params['poses']
                if isinstance(cur_poses, list):
                    cur_poses = np.array(cur_poses)
                else:
                    cur_poses = cur_poses.copy()

                # Kiểm tra kích thước
                if len(src_poses) >= 63 and len(cur_poses) >= 63:
                    cur_poses[0:69] = src_poses[0:69]
                    current_params['poses'] = cur_poses
                    print(f"✓ Replaced arm joints (idx 51-62) for {base_name}")
                else:
                    print(f"⚠ Warning: poses vector too short in {src_json}, skipping replacement")
            else:
                print(f"⚠ Warning: 'poses' not found in {src_json}")
        else:
            print(f"⚠ Warning: source file not found at {src_json}, keeping original poses")

        # Lưu params đã fix
        output_path = join(self.output_dir, f'{base_name}.json')
        params_to_save = {}
        for key, value in current_params.items():
            if isinstance(value, np.ndarray):
                params_to_save[key] = value.tolist()
            elif isinstance(value, list):
                params_to_save[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v
                    for v in value
                ]
            else:
                params_to_save[key] = value

        with open(output_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)

        return current_params

    def __call__(self, params, imgnames, **kwargs):
        """
        Args:
            params:   dict hoặc list[dict] – {'Rh', 'Th', 'poses', 'shapes', ...}
            imgnames: str hoặc list[str]
            **kwargs: các key khác (cameras, imgnames, ...) được pass-through

        Returns:
            dict với 'params' đã fix + tất cả kwargs
        """
        if isinstance(params, dict):
            params_list   = [params]
            imgnames_list = [imgnames] if isinstance(imgnames, str) else imgnames
        else:
            params_list   = params
            imgnames_list = imgnames if isinstance(imgnames, list) else [imgnames]

        fixed_params_list = [
            self.fix_single_param(p, n)
            for p, n in zip(params_list, imgnames_list)
        ]

        result = kwargs.copy()
        result['params'] = fixed_params_list[0] if isinstance(params, dict) else fixed_params_list
        return result