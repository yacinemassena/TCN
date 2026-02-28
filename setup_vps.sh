#!/bin/bash
# VPS Setup Script for TCN Pretraining
# Run this on a fresh VPS with CUDA-enabled GPU

set -e  # Exit on error

echo "=========================================="
echo "TCN Pretraining VPS Setup"
echo "=========================================="

# 1. Upgrade PyTorch to latest version
echo "[1/6] Upgrading PyTorch..."
pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 2. Clone repository
echo "[2/6] Cloning TCN repository..."
cd /
if [ -d "/TCN" ]; then
    echo "TCN directory exists, pulling latest..."
    cd /TCN
    git pull
else
    git clone https://github.com/yacinemassena/TCN.git
    cd /TCN
fi

# 3. Install Python dependencies
echo "[3/6] Installing dependencies..."
pip install pandas pyarrow tqdm boto3

# 4. Download RV file from R2
echo "[4/6] Downloading RV file from R2..."
python3 << 'EOF'
import boto3
from pathlib import Path

s3 = boto3.client('s3',
    endpoint_url='https://2a139e9393f803634546ad9d541d37b9.r2.cloudflarestorage.com',
    aws_access_key_id='fdfa18bf64b18c61bbee64fda98ca20b',
    aws_secret_access_key='394c88a7aaf0027feabe74ae20da9b2f743ab861336518a09972bc39534596d8'
)

Path('/TCN/datasets/2022-2023').mkdir(parents=True, exist_ok=True)
s3.download_file('europe', 'datasets/spy_daily_rv.parquet', '/TCN/datasets/2022-2023/spy_daily_rv.parquet')
print('Downloaded spy_daily_rv.parquet')
EOF

# 5. Download index_data from R2
echo "[5/6] Downloading index_data from R2..."
python3 download_index_data.py

# 6. Verify setup
echo "[6/6] Verifying setup..."
python3 -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"None\"}')"
ls -lh /TCN/datasets/2022-2023/

echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo ""
echo "To start training, run:"
echo "  cd /TCN"
echo "  python pretrain_tcn_rv.py --profile h100 --stream index --epochs 100 --no-checkpoint"
echo ""
echo "Available profiles: rtx5080 (16GB), h100 (80GB), a100 (80GB), amd (192GB)"
echo "Add --no-checkpoint flag to disable gradient checkpointing (faster, more VRAM)"
echo ""
