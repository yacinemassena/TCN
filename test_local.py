"""Run local training on test dataset for comparison."""
import sys
from pathlib import Path

# Override paths before importing config
import config_pretrain
config_pretrain.STREAM_CONFIGS['index'].data_path = str(Path(__file__).parent / 'test_dataset' / 'index_data')

# Now run the main script
from pretrain_tcn_rv import main
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--profile', type=str, default='rtx5080')
parser.add_argument('--stream', type=str, default='index')
parser.add_argument('--epochs', type=int, default=1)
parser.add_argument('--rv_file', type=str, default=str(Path(__file__).parent / 'test_dataset' / 'spy_daily_rv.parquet'))
parser.add_argument('--device', type=str, default='cuda')

args = parser.parse_args()
main(args)
