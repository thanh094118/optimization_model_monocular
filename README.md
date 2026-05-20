#ver demo test - chỉ hỗ trợ chạy thủ công từng testcase  
Input là 2 file .pkl từ wham/wham_opencap  
#cách setup  - cài sẵn py3.9 + pip ...  
chmod +x setup.sh  
./setup.sh  
mô tả: cài dependent + tải file cần thiết + data để chạy demo  
#cách chạy  
chạy flow mặc định:  
- python main.py --config configs/pipeline.yml
chạy từng pipeline
- python main.py --config configs/pipeline.yml  --stage u_choice  
các option hỗ trợ:  
            "all",  
            "pose",  
            "fusion",  
            "learnable",  
            "visualization",  
            "pose_fusion",  
            "postprocess",  
            "refirement",  
