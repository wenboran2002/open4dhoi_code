# 4DHOISolver

Obtain `upload_records.json`(TODO) and put it under the directory.
### SMPL Model
Download `SMPLX_NEUTRAL.npz` from [its website](https://smpl-x.is.tue.mpg.de/download.php) and put it under `video_optimizer/smpl_model`
### optimize
Run `python optimize.py` to optimize all records **with annotation progress 4**  
### render
Run this command to visualize the optimize result under global camera
```shell
python render.py --data_dir [path to record]
```




