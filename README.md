<div align="center">

Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction

[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b.svg)](https://arxiv.org/abs/2512.00960)
[![Project Page](https://img.shields.io/badge/Project-Page-blue)](https://wenboran2002.github.io/open4dhoi/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Dataset-yellow)](https://huggingface.co/datasets/wenboran2002/Open4DHOI)

</div>

<!-- 建议在这里放一张 teaser 图片或 GIF 动图，展示你的核心效果 -->

<p align="center">
<img src="assets/teaser.jpg" alt="Teaser Image" width="400"/>





</p>


## 📰 News

<!-- 记录项目的更新日志 -->

[2025-12-02] Annotate app code released!

<!-- [2025-12-03] 4DHOISolver code released! -->

## 🚀 To Do

[x] Release core inference code.

[ ] Release Automatic 4DHOI Reconstruction Code.

[ ] Release Dataset

## 🛠️ Installation


```bash
conda env create -n 4dhoi_solver python=3.10
conda activate 4dhoi_solver
pip install -r requirements.txt
```

## 🖥️  Annotate app

### Data Preparation
You can download the test data from [Google Drive](https://drive.google.com/uc?export=download&id=1a9iUSfuuBrB2q6iewi4uxMAB9XIrvuJo) and place it in ./demo.

The data structure should be like this:
```
./demo
├── align ## depth alignment result for initialization
├── motion ## motion reconstruction from GVHMR
├── video 
└── obj_org.obj ## object model
```

### Install
please follow https://github.com/facebookresearch/co-tracker to install co-tracker. Remember to download scaled_online.pth from co-tracker and place it in Annot-app/co-tracker/checkpoints/

Then install the Annot-app code:

```
cd Annot-app/co-tracker
pip install -e .
```

### Usage
See `Annot-app/co-tracker/README.md` for more details.






## 📖 Citation

If you find this code useful for your research, please consider citing our paper:

<!-- 替换为你的 BibTeX -->
```
@misc{wen2025efficientscalablemonocularhumanobject,
      title={Efficient and Scalable Monocular Human-Object Interaction Motion Reconstruction}, 
      author={Boran Wen and Ye Lu and Keyan Wan and Sirui Wang and Jiahong Zhou and Junxuan Liang and Xinpeng Liu and Bang Xiao and Dingbang Huang and Ruiyang Liu and Yong-Lu Li},
      year={2025},
      eprint={2512.00960},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2512.00960}, 
}
```