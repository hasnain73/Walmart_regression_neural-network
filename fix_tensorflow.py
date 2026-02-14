"""
Auto-fix script for TensorFlow compatibility
Fixes the DLL load error by installing the correct TensorFlow version
"""

import subprocess
import sys

print("="*60)
print("TensorFlow Compatibility Fix")
print("="*60)
print(f"\nPython Version: {sys.version}")

print("\n[1/3] Uninstalling old TensorFlow...")
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "tensorflow"])

print("\n[2/3] Installing TensorFlow 2.16.1 (Python 3.12 compatible)...")
subprocess.run([sys.executable, "-m", "pip", "install", "tensorflow==2.16.1"])

print("\n[3/3] Installing updated dependencies...")
subprocess.run([sys.executable, "-m", "pip", "install", "numpy==1.26.4", "flask==3.0.0"])

print("\n" + "="*60)
print("✓ Fix Complete!")
print("="*60)
print("\nNow you can run:")
print("  python app_flask.py")
print("  python app.py")
print("  python train_model.py")
print("="*60)
