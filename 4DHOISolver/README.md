
### 🎬 Usage


Run optimization on any record:

```bash
python optimize.py --data_dir [data]
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








