# 🎯 4DHOISolver

4DHOISolver is a tool for optimizing and rendering 4D human-object interaction sequences. It takes annotated video data and produces optimized human body parameters and object poses in a global coordinate system.

## 🚀 Quick Start

### 📦 Environment Setup

```bash
conda env create -f environment.yml
conda activate 4dhoi_solver
cd multiperson/sdf && pip install -e . --no-build-isolation && cd ../..
cd multiperson/neural_renderer && pip install -e . --no-build-isolation && cd ../..
```

### 🔑 Download SMPL-X Model

1. Download `SMPLX_NEUTRAL.npz` from the [SMPL-X website](https://smpl-x.is.tue.mpg.de/download.php)
```bash
mkdir -p video_optimizer/smpl_models
mv SMPLX_NEUTRAL.npz video_optimizer/smpl_models/
```

### 🎬 Usage


Run optimization on all records with annotation progress 4:

```bash
python optimize.py
```

**Output structure:**
```
session_folder/
├── final_optimized_parameters/
│   ├── all_parameters_YYYYMMDD_HHMMSS.json
│   ├── transformed_parameters_YYYYMMDD_HHMMSS.json
│   └── transformed_object_YYYYMMDD_HHMMSS.obj
├── optimized_frames/
│   └── *.png
└── optimized_preview.mp4
```

### 2️⃣ Render Visualization

Visualize the optimized results under a global camera view:

```bash
python render.py --data_dir [path/to/session/folder]
```
It saves Rendered video to `[data_dir]/output_render.mp4`








