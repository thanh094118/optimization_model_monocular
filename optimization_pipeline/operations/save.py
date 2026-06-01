"""
myeasymocap/operations/save.py
==============================
Module lưu SMPL parameters ra folder params dưới dạng JSON.

YAML:
    save_params:
      module: myeasymocap.operations.save.SaveParams
      key_from_data: [imgnames]
      key_from_previous: [params]
      args:
        output_dir: output/params

Nếu bước write bị lỗi 'unexpected keyword argument meta',
dùng WriteSMPLWrapper thay cho myeasymocap.io.write.WriteSMPL:

    write:
      module: myeasymocap.operations.save.WriteSMPLWrapper
      key_from_data: [meta]
      key_from_previous: [params, model]
      args:
        name: smpl
        output: output/smpl_results
"""

import os
import json
import numpy as np
from os.path import join


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _stem(imgname: str) -> str:
    return os.path.splitext(os.path.basename(imgname))[0]


def _flatten_imgnames(imgnames, n_frames: int):
    if isinstance(imgnames, str):
        return [imgnames]
    flat = []
    for item in imgnames:
        if isinstance(item, (list, tuple)):
            flat.extend(item)
        else:
            flat.append(item)
    while flat and len(flat) < n_frames:
        flat.append(flat[-1])
    return flat[:n_frames]


def _to_serializable(obj):
    if hasattr(obj, 'detach'):
        obj = obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    return obj


# ---------------------------------------------------------------------------
# SaveParams – lưu params sau bước infer
# ---------------------------------------------------------------------------
class SaveParams2:
    """
    Lưu params của từng frame thành <output_dir>/<stem(imgname)>.json.
    key_from_data: [imgnames, params]
    """

    def __init__(self, output_dir: str = 'output/params', **kwargs):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_single(self, param: dict, imgname):
        if isinstance(imgname, (list, tuple)):
            imgname = imgname[0]
        base_name = _stem(imgname)

        params_to_save = {}
        for key, value in param.items():
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                params_to_save[key] = value.tolist()
            elif isinstance(value, list):
                params_to_save[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v
                    for v in value
                ]
            else:
                params_to_save[key] = value

        out_path = join(self.output_dir, f'{base_name}.json')
        with open(out_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        print(f'[SaveParams] Saved -> {out_path}')

    def __call__(self, imgnames, params, **kwargs):
        # params có thể là dict (1 người) hoặc list[dict] (nhiều người)
        if isinstance(params, dict):
            params_list = [params]
            imgnames_list = [imgnames] if isinstance(imgnames, str) else imgnames
        else:
            params_list = params
            imgnames_list = imgnames if isinstance(imgnames, list) else [imgnames]

        for param, imgname in zip(params_list, imgnames_list):
            self._save_single(param, imgname)

        # pass-through toàn bộ, không thay đổi gì
        return {**kwargs, 'params': params, 'imgnames': imgnames}

class SaveParams:
    """
    Lưu params của từng frame thành <output_dir>/<stem(imgname)>.json.
    Trả về {**kwargs, 'params': params} để pipeline tiếp tục bình thường.
    """

    def __init__(self, output_dir: str = 'output/params', **kwargs):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _save_single(self, param: dict, imgname):
        if isinstance(imgname, (list, tuple)):
            imgname = imgname[0]
        base_name = _stem(imgname)

        params_to_save = {}
        for key, value in param.items():
            if hasattr(value, 'detach'):
                value = value.detach().cpu().numpy()
            if isinstance(value, np.ndarray):
                params_to_save[key] = value.tolist()
            elif isinstance(value, list):
                params_to_save[key] = [
                    v.tolist() if isinstance(v, np.ndarray) else v
                    for v in value
                ]
            else:
                params_to_save[key] = value

        out_path = join(self.output_dir, f'{base_name}.json')
        with open(out_path, 'w') as f:
            json.dump(params_to_save, f, indent=2)
        print(f'[SaveParams] Saved -> {out_path}')

    def __call__(self, params, imgnames, **kwargs):
        if isinstance(params, dict):
            params_list   = [params]
            imgnames_list = [imgnames] if isinstance(imgnames, str) else imgnames
        else:
            params_list   = params
            imgnames_list = imgnames if isinstance(imgnames, list) else [imgnames]

        for param, imgname in zip(params_list, imgnames_list):
            self._save_single(param, imgname)

        result = kwargs.copy()
        result['params'] = params
        return result


# ---------------------------------------------------------------------------
# WriteSMPLWrapper – bọc WriteSMPL gốc, absorb **kwargs thừa
# ---------------------------------------------------------------------------

class WriteSMPLWrapper:
    """
    Wrapper của myeasymocap.io.write.WriteSMPL.
    Nhận **kwargs từ pipeline (kể cả 'meta', 'imgnames', ...)
    rồi chỉ truyền đúng những gì WriteSMPL cần.

    Dùng thay cho myeasymocap.io.write.WriteSMPL trong YAML:

        write:
          module: myeasymocap.operations.save.WriteSMPLWrapper
          key_from_data: [meta]
          key_from_previous: [params, model]
          args:
            name: smpl
            output: output/smpl_results
    """

    def __init__(self, name: str = 'smpl', output: str = '/tmp',
                 write_vertices: bool = False, **kwargs):
        # Import ở đây để tránh circular import lúc load module
        from myeasymocap.io.write import WriteSMPL
        self._inner = WriteSMPL(
            name=name,
            output=output,
            write_vertices=write_vertices,
        )

    def __call__(self, params=None, results=None, meta=None,
                 model=None, **kwargs):
        # Chỉ truyền đúng 4 args mà WriteSMPL.__call__ biết
        return self._inner(
            params=params,
            results=results,
            meta=meta,
            model=model,
        )