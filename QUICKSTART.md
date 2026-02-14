# Quick Start Guide

## 🚀 For Google Colab (Fastest)

1. **Open Colab Notebook:**
   - Upload `walmart_sales_prediction.ipynb` to Google Colab
   - Or open directly: [Google Colab](https://colab.research.google.com)

2. **Upload kaggle.json:**
   - Get from Kaggle Account Settings → Create New API Token
   - Upload when prompted in notebook

3. **Run All Cells:**
   - Runtime → Run All
   - Wait ~5 minutes for completion

4. **Download Files:**
   - model.h5, scaler.pkl, app.py, requirements.txt
   - Ready for Hugging Face deployment!

---

## 💻 For Local Development

### Windows (Easiest)
```cmd
# Double-click this file:
setup_env.bat

# Then download Walmart.csv from Kaggle
# Place in walmart folder

# Train model
walmart_nn_env\Scripts\activate
python train_model.py

# Run app
python app.py
```

### Manual Setup
```bash
# Create environment
python -m venv walmart_nn_env

# Activate
walmart_nn_env\Scripts\activate  # Windows
source walmart_nn_env/bin/activate  # Mac/Linux

# Install
pip install -r requirements.txt

# Download dataset
# Place Walmart.csv in project folder

# Train
python train_model.py

# Run app
python app.py
```

---

## 🤗 Deploy to Hugging Face

1. **Login:** [huggingface.co](https://huggingface.co)
2. **Create Space:** Choose Gradio SDK
3. **Upload Files:**
   - app.py
   - requirements.txt
   - models/model.h5
   - models/scaler.pkl
   - models/feature_names.pkl
4. **Wait for build** (~2 min)
5. **Share your link!**

---

## 📤 Push to GitHub

```bash
# Init repo
git init
git add .
git commit -m "Walmart Sales NN Predictor"

# Add remote (replace YOUR_USERNAME)
git remote add origin https://github.com/YOUR_USERNAME/walmart-sales-nn.git

# Push
git branch -M main
git push -u origin main
```

---

## 📁 Project Files

### Essential Files:
- `train_model.py` - Training script
- `app.py` - Gradio interface
- `requirements.txt` - Dependencies
- `walmart_sales_prediction.ipynb` - Colab notebook

### Documentation:
- `README.md` - Full documentation
- `DEPLOYMENT_GUIDE.md` - Detailed deployment steps
- `QUICKSTART.md` - This file

### Setup:
- `setup_env.bat` - Auto setup (Windows)
- `.gitignore` - Git exclusions

### Generated (after training):
- `models/model.h5` - Trained model
- `models/scaler.pkl` - Feature scaler
- `models/feature_names.pkl` - Feature list
- `training_results.png` - Visualization

---

## ⚡ Quick Commands

```bash
# Setup
setup_env.bat

# Train
python train_model.py

# Test locally
python app.py

# Git push
git add . && git commit -m "update" && git push
```

---

## 🎯 Submission Checklist

- [ ] Code runs in Colab
- [ ] Model trained successfully
- [ ] Gradio app works locally
- [ ] Pushed to GitHub
- [ ] Deployed to Hugging Face
- [ ] Links documented

---

**Time to Complete:** ~15-20 minutes

**For Help:** Check README.md or DEPLOYMENT_GUIDE.md
