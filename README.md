<div align="center">

Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction




</div>

<!-- 建议在这里放一张 teaser 图片或 GIF 动图，展示你的核心效果 -->

<p align="center">
<img src="assets/teaser.jpg" alt="Teaser Image" width="400"/>





</p>


📰 News

<!-- 记录项目的更新日志 -->

[2025-12-02] Automatic 4d reconstruction code released!

[2025-12-03] 4DHOISolver code released!

🚀 To Do

[x] Release core inference code.

[ ] Release App

[ ] Release Dataset

🛠️ Installation


```
# Clone this repository
git clone [https://github.com/username/repo-name.git](https://github.com/username/repo-name.git)
cd repo-name

# Create environment (Modify requirements.txt as needed)
conda create -n es_hmr python=3.9
conda activate es_hmr

# Install PyTorch (Adjust cuda version according to your GPU)
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

# Install other dependencies
pip install -r requirements.txt

# (Optional) Install third-party libraries (e.g., SMPL-X, PyTorch3D)
# pip install "git+[https://github.com/facebookresearch/pytorch3d.git](https://github.com/facebookresearch/pytorch3d.git)"
```

📂 Data Preparation

Please download the datasets and arrange them in the data/ folder structure as follows:
```
<!-- 描述你的数据文件结构 -->

repo-name/
|-- data/
|   |-- BEHAVE/
|   |   |-- images/
|   |   |-- annotations/
|   |-- InterCap/
|   |-- smpl_models/
|       |-- smplx/
```

SMPL/SMPL-X Models: Please register and download from the official website.

Datasets: Download BEHAVE or InterCap from their respective sources.

🏃 Auto-Reconstruction / Demo

To run reconstruction on a sample video:
```
python demo.py \
    --input_path assets/sample_video.mp4 \
    --checkpoint checkpoints/model_sota.pth \
    --output_dir results/demo_output \
    --visualize
```

📖 Citation

If you find this code useful for your research, please consider citing our paper:

<!-- 替换为你的 BibTeX -->
```

```