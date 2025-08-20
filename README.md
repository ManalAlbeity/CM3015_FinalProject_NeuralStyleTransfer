# UrbanBrush: Neural Style Transfer for Cityscapes (Final Year Project)
## CM3015 Template: Neural Style Transfer

This project explores **Neural Style Transfer (NST)** using three approaches:
- **Gatys et al. (2015)** — Classical optimisation-based NST
- **TF-Hub Johnson Model** — Pretrained fast NST
- **AdaIN (Adaptive Instance Normalisation)** — Real-time style transfer


## Features
- Batch stylisation of content × style pairs  
- α:β ratio exploration (content vs. style balance)  
- Animated transitions (GIFs)  
- Video style transfer with multiple models  
- Quantitative evaluation (SSIM, LPIPS, ExecTime)  
- Peer feedback + interactive sliders (Jupyter)  
- **Streamlit Web App**: Upload content/style, apply NST, and optionally **mask foreground objects** for selective stylisation  


## Folder Structure
- `Final_NST_Project.ipynb` — Main notebook (GPU research pipeline)    
- `nst_app/app.py` — Streamlit web application  
- `requirements_notebook.txt` — Dependencies for the notebook (GPU, research)  
- `requirements_app.txt` — Dependencies for the app (CPU, deployment)  
- `output/images/` — Stylised results  
- `output/gifs/` — Transition GIFs  
- `output/videos/` — Side-by-side comparisons  
- `output/batch/` — Batch outputs + results.csv


 ## Technical Notes

 ### Separation of Environments
- **GPU environment (Research Notebook):** Ensures smooth model training and evaluation.  
- **CPU environment (Streamlit App):** Designed for maximum accessibility so users without GPUs can still run the app.

### Extra Step Beyond Standard NST
Originally, the aim was **full-image stylisation**.  
We extended this work by integrating **object detection masks**, enabling **selective stylisation** of just the **foreground object** in an image.

### Why AdaIN?
- **Johnson model:** Fast but limited to fixed styles.  
- **Gatys model:** Very flexible but extremely slow.  
- **AdaIN:** Combines **speed (near real-time)** with **flexibility (arbitrary styles)**, making it ideal for an **interactive app**.

### Deployment
- **Notebook:** Runs on GPU workstation or Google Colab.  
- **App:** Can be deployed via **Streamlit Cloud (free hosting)** or run locally on CPU.
- 

## Installation

Clone this repo:
```bash
git clone https://github.com/ManalAlbeity/Neural-Style-Transfer-Project.git
cd Neural-Style-Transfer-Project
```

## 1. Notebook Environment (GPU-enabled for research)

This environment is optimized for training and experimentation.
```bash
pip install -r requirements_notebook.txt
```
or directly in your environment install:
```bash
pip install tensorflow==2.10.0 torch==2.7.1+cu118 torchvision==0.22.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118 opencv-python==4.12.0 scikit-image imageio Pillow matplotlib==3.10.5 ipywidgets==8.1.7 numpy==1.26.4 pandas tensorflow-hub==0.16.1 lpips seaborn
```

## 2. App Environment (CPU-friendly for deployment)

This environment is optimized for running the Streamlit demo on most machines.
```bash
pip install -r requirements_app.txt
```

## Usage

**Notebook**

Run the notebook for experiments and research workflow:
```bash
jupyter notebook Final_NST_Project.ipynb
```

**Streamlit App**

Launch the interactive web app:
```bash
streamlit run app.py
```
