# Neural Style Transfer — Streamlit App 

This folder contains a beginner-friendly **Streamlit web app** for Neural Style Transfer (NST).  
The app supports three model options:

1. **TF-Hub (Fast)** — works out-of-the-box (Magenta arbitrary style).
2. **Gatys (Classic)** — slower, runs iterative optimisation in TensorFlow.
3. **AdaIN (Optional)** — requires two extra weight files in `adain/`:
   - `adain/vgg_normalised.pth`
   - `adain/decoder.pth`

> If you don’t add AdaIN weights, the app will still work fine with **TF-Hub** and **Gatys**.

## How to run locally (optional)

1. Create a virtual environment (Windows PowerShell):
   ```pwsh
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   (macOS/Linux):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. Install packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the app:
   ```bash
   streamlit run app.py
   ```

Good luck and have fun!

