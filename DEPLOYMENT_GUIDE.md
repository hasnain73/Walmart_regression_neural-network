# 🚀 Complete Deployment Guide

## 📋 Table of Contents
1. [Local Setup](#local-setup)
2. [Training the Model](#training-the-model)
3. [Testing Locally](#testing-locally)
4. [GitHub Deployment](#github-deployment)
5. [Hugging Face Deployment](#hugging-face-deployment)

---

## 🖥️ Local Setup

### Prerequisites
- Python 3.10.19 installed
- Internet connection
- Kaggle account (for dataset download)

### Step 1: Setup Environment

**Windows:**
```bash
setup_env.bat
```

This will:
- Create `walmart_nn_env` virtual environment
- Install all required packages
- Set up the project

**Manual Alternative:**
```bash
python -m venv walmart_nn_env
walmart_nn_env\Scripts\activate
pip install -r requirements.txt
```

---

## 📊 Training the Model

### Step 1: Get the Dataset

1. **Create Kaggle Account** (if you don't have one)
   - Go to [kaggle.com](https://www.kaggle.com)
   - Sign up for free

2. **Download Dataset**
   - Visit: https://www.kaggle.com/datasets/mikhail1681/walmart-sales
   - Click **"Download"** button
   - Extract `Walmart.csv`
   - Place in project root: `walmart/Walmart.csv`

### Step 2: Train the Model

```bash
# Activate environment (if not already active)
walmart_nn_env\Scripts\activate

# Run training
python train_model.py
```

**Training Time:** ~1-2 minutes (depending on your CPU)

**Output Files:**
- `models/model.h5` - Trained neural network
- `models/scaler.pkl` - Feature scaler
- `models/feature_names.pkl` - Feature names
- `training_results.png` - Visualization

---

## 🧪 Testing Locally

### Run Gradio App

```bash
python app.py
```

**Expected Output:**
```
Loading model and preprocessors...
Model loaded successfully!
Features expected: [...]
Number of features: X

========================================
🚀 Launching Gradio App...
========================================
Running on local URL:  http://127.0.0.1:7860
```

### Test the Interface
1. Open browser: `http://localhost:7860`
2. Enter feature values
3. Click **Submit**
4. View prediction

**✅ If it works locally, you're ready to deploy!**

---

## 📤 GitHub Deployment

### First-Time Setup

#### Step 1: Create GitHub Repository
1. Go to [github.com](https://github.com)
2. Click **"New Repository"**
3. Repository details:
   - **Name:** `walmart-sales-nn`
   - **Description:** "Neural Network for Walmart Sales Prediction"
   - **Public** (for Hugging Face integration)
   - Don't initialize with README (we have one)
4. Click **"Create Repository"**

#### Step 2: Push Code to GitHub

```bash
# Navigate to project
cd c:\Users\student\Desktop\walmart

# Initialize git
git init

# Add all files
git add .

# Commit
git commit -m "Initial commit: Walmart Sales NN Predictor"

# Add your GitHub repository as remote
# Replace YOUR_USERNAME with your GitHub username
git remote add origin https://github.com/YOUR_USERNAME/walmart-sales-nn.git

# Push to GitHub
git branch -M main
git push -u origin main
```

**Note:** If git asks for authentication:
- Use GitHub Personal Access Token (not password)
- Generate token: GitHub Settings → Developer Settings → Personal Access Tokens

### Update Code Later

```bash
git add .
git commit -m "Updated model performance"
git push
```

---

## 🤗 Hugging Face Deployment

### Step 1: Create Hugging Face Account

1. Go to [huggingface.co/join](https://huggingface.co/join)
2. Sign up (free)
3. Verify email

### Step 2: Create New Space

1. **Go to Spaces**
   - Visit: https://huggingface.co/spaces
   - Click **"Create new Space"**

2. **Configure Space**
   - **Owner:** Your username
   - **Space name:** `walmart-sales-predictor`
   - **License:** Apache 2.0
   - **Select SDK:** **Gradio**
   - **Space hardware:** CPU basic (free)
   - **Visibility:** Public

3. Click **"Create Space"**

### Step 3: Upload Files

**Method A: Web Interface (Easiest)**

1. In your Space, click **"Files"** tab
2. Click **"Add file"** → **"Upload files"**
3. Upload these files:
   ```
   app.py
   requirements.txt
   ```
4. Create `models` folder:
   - Click **"Add file"** → **"Create a new file"**
   - Path: `models/.gitkeep`
   - Commit
5. Upload model files to `models/` folder:
   ```
   models/model.h5
   models/scaler.pkl
   models/feature_names.pkl
   ```

**Method B: Git Clone (Advanced)**

```bash
# Install git-lfs first
git lfs install

# Clone your space
git clone https://huggingface.co/spaces/YOUR_USERNAME/walmart-sales-predictor
cd walmart-sales-predictor

# Copy files
copy ..\app.py .
copy ..\requirements.txt .
mkdir models
copy ..\models\*.* models\

# Add and commit
git add .
git commit -m "Deploy Walmart Sales Predictor"
git push
```

### Step 4: Verify Deployment

1. **Wait for Build**
   - Hugging Face will automatically build your app
   - Check **"Logs"** tab for progress
   - Build time: ~2-3 minutes

2. **Test Your App**
   - Once build succeeds, **"App"** tab will show your interface
   - Test with sample inputs
   - Verify predictions work

3. **Your Live URL:**
   ```
   https://huggingface.co/spaces/YOUR_USERNAME/walmart-sales-predictor
   ```

### Step 5: Share Your Project

✅ **GitHub Repo:** `https://github.com/YOUR_USERNAME/walmart-sales-nn`

✅ **Live Demo:** `https://huggingface.co/spaces/YOUR_USERNAME/walmart-sales-predictor`

---

## 🎯 Submission Checklist

For academic submission, ensure you have:

- [ ] **Code Repository**
  - [ ] GitHub repo is public
  - [ ] All code files included
  - [ ] README.md is complete
  - [ ] .gitignore is configured

- [ ] **Model Training**
  - [ ] Model trained successfully
  - [ ] Training results saved
  - [ ] Evaluation metrics documented

- [ ] **Deployment**
  - [ ] Gradio app works locally
  - [ ] Hugging Face Space is live
  - [ ] App is accessible publicly

- [ ] **Documentation**
  - [ ] README with instructions
  - [ ] Code has comments
  - [ ] Deployment guide included

- [ ] **Deliverables**
  - [ ] GitHub repo link
  - [ ] Hugging Face Space link
  - [ ] Training visualization
  - [ ] Model evaluation metrics

---

## 🐛 Troubleshooting

### Issue: "Module not found"
```bash
# Reactivate environment and reinstall
walmart_nn_env\Scripts\activate
pip install -r requirements.txt
```

### Issue: "Cannot find Walmart.csv"
- Ensure `Walmart.csv` is in project root
- Check filename (case-sensitive on some systems)

### Issue: Hugging Face build fails
- Check `requirements.txt` versions
- Verify all model files uploaded
- Check build logs in HF Space

### Issue: Git authentication fails
- Use Personal Access Token instead of password
- GitHub Settings → Developer Settings → Personal Access Tokens → Generate new token

---

## 📞 Support Resources

- **TensorFlow Docs:** [tensorflow.org/api_docs](https://www.tensorflow.org/api_docs)
- **Gradio Docs:** [gradio.app/docs](https://www.gradio.app/docs/)
- **Hugging Face Docs:** [huggingface.co/docs/hub/spaces](https://huggingface.co/docs/hub/spaces)
- **GitHub Docs:** [docs.github.com](https://docs.github.com)

---

**Good luck with your submission! 🚀**

*Last updated: 2026-02-13*
