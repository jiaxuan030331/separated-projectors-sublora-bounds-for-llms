@echo off
echo ============================================
echo SubLoRA Windows Environment Setup
echo ============================================
echo.

echo Step 1: Removing any existing sublora environment...
call conda env remove -n sublora -y
echo.

echo Step 2: Creating new conda environment with Python 3.8.17...
call conda create -n sublora python=3.8.17 pip -y
if errorlevel 1 (
    echo ERROR: Failed to create conda environment
    exit /b 1
)
echo.

echo Step 3: Activating sublora environment...
call conda activate sublora
if errorlevel 1 (
    echo ERROR: Failed to activate environment
    exit /b 1
)
echo.

echo Step 4: Installing PyTorch 2.0.1 with CUDA 11.8 support...
pip install torch==2.0.1 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
if errorlevel 1 (
    echo ERROR: Failed to install PyTorch
    exit /b 1
)
echo.

echo Step 5: Installing core training dependencies...
pip install transformers==4.31.0 datasets==2.14.2 accelerate==0.21.0 peft==0.4.0 loralib==0.1.1 evaluate==0.4.0 wandb==0.15.3 hydra-core==1.3.2 einops==0.6.1 sentencepiece==0.1.99
if errorlevel 1 (
    echo ERROR: Failed to install core dependencies
    exit /b 1
)
echo.

echo Step 6: Installing scientific computing packages...
pip install numpy==1.24.4 scipy==1.10.1 scikit-learn==1.2.2 pandas==2.0.3
if errorlevel 1 (
    echo ERROR: Failed to install scientific packages
    exit /b 1
)
echo.

echo Step 7: Installing the sublora package in editable mode...
pip install -e .
if errorlevel 1 (
    echo ERROR: Failed to install sublora package
    exit /b 1
)
echo.

echo ============================================
echo Installation Complete!
echo ============================================
echo.
echo To activate the environment, run:
echo     conda activate sublora
echo.
echo To verify the installation, run:
echo     python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
echo.