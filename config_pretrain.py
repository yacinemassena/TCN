"""
Configuration for TCN Pretraining on SPY Realized Volatility.
"""

from dataclasses import dataclass, field
from typing import Optional
from pathlib import Path

# Project root directory (where this config file lives)
PROJECT_ROOT = Path(__file__).parent.resolve()


@dataclass
class GPUProfile:
    """GPU profile for VRAM-aware batching."""
    name: str = 'rtx5080'
    vram_gb: int = 16
    max_chunks_per_batch: int = 2000      # 16GB: 2000, 80GB: 11600
    filter_stocks: bool = True            # True for 16GB, False for 80GB
    top_n_stocks: int = 100               # 0 = all stocks
    prefetch_files: int = 8               # Files to keep in RAM


# Predefined GPU profiles
GPU_PROFILES = {
    'rtx5080': GPUProfile(
        name='rtx5080',
        vram_gb=16,
        max_chunks_per_batch=2000,
        filter_stocks=True,
        top_n_stocks=100,
        prefetch_files=8,
    ),
    'h100': GPUProfile(
        name='h100',
        vram_gb=80,
        max_chunks_per_batch=11600,
        filter_stocks=False,
        top_n_stocks=0,
        prefetch_files=16,
    ),
    'a100': GPUProfile(
        name='a100',
        vram_gb=80,
        max_chunks_per_batch=11600,
        filter_stocks=False,
        top_n_stocks=0,
        prefetch_files=16,
    ),
}


@dataclass
class StreamConfig:
    """Configuration for a single data stream with VRAM-aware batch sizing."""
    name: str
    data_path: str
    filter_tickers: bool = False
    allowed_tickers_file: Optional[str] = None
    num_tickers: int = 0  # For ticker embedding (0 = disabled)
    # Batch sizing per GPU (chunks per batch) - tuned to tick volume
    # Stocks (filtered): ~8M ticks/day → baseline
    # Options: ~5.4M ticks/day → 0.7x stocks
    # Index: ~347K ticks/day → 0.04x stocks (can fit more frames per batch)
    max_chunks_16gb: int = 2000   # RTX 5080
    max_chunks_80gb: int = 11600  # H100/A100
    prefetch_files: int = 8


# Predefined stream configurations with tuned batch sizes
STREAM_CONFIGS = {
    'stocks': StreamConfig(
        name='stocks',
        data_path=str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'polygon_stock_trades'),
        filter_tickers=True,  # Filter to top 100 on 16GB
        allowed_tickers_file=str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'top_100_stocks.txt'),
        num_tickers=100,
        max_chunks_16gb=2000,   # ~8M ticks/day filtered → baseline
        max_chunks_80gb=11600,
        prefetch_files=8,
    ),
    'options': StreamConfig(
        name='options',
        data_path=str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'options_trades'),
        filter_tickers=False,
        allowed_tickers_file=None,
        num_tickers=0,
        max_chunks_16gb=2800,   # ~5.4M ticks/day → more headroom
        max_chunks_80gb=16000,
        prefetch_files=12,
    ),
    'index': StreamConfig(
        name='index',
        data_path=str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'index_data'),
        filter_tickers=False,
        allowed_tickers_file=None,
        num_tickers=0,
        max_chunks_16gb=1500,   # Reduced from 4000 - RTX 5080 OOM with TCN-12
        max_chunks_80gb=24000,
        prefetch_files=16,
    ),
}


@dataclass
class PretrainDataConfig:
    """Dataset configuration for pretraining."""
    # Data paths (multi-stream) - relative to PROJECT_ROOT
    stocks_path: str = str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'polygon_stock_trades')
    options_path: str = str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'options_trades')
    index_path: str = str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'index_data')
    rv_file: str = str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'spy_daily_rv.parquet')
    top_stocks_file: str = str(PROJECT_ROOT / 'datasets' / '2022-2023' / 'top_100_stocks.txt')
    
    # Frame settings
    frame_interval: str = '10s'
    chunk_len: int = 256
    num_frames: int = 360  # 1 hour of 10s frames per sample
    
    # RV target
    rv_horizon_days: int = 30  # 30-day forward RV
    
    # Split dates
    train_end: str = '2022-12-31'
    val_end: str = '2023-06-30'
    # test: everything after val_end
    
    # Features
    dim_in: int = 3  # price, size, dt
    weight_mode: str = 'tick_count'
    
    # Batching (set by GPU profile)
    max_chunks_per_batch: int = 2000
    prefetch_files: int = 8


@dataclass
class PretrainTCNConfig:
    """TCN encoder configuration."""
    dim_in: int = 3
    hidden_dim: int = 512
    num_layers: int = 12
    kernel_size: int = 3
    dropout: float = 0.1
    checkpoint_every: int = 0  # 0 = no checkpointing
    
    # Ticker embedding (disabled for SPY-only pretraining)
    num_tickers: int = 0
    ticker_embed_dim: int = 16


@dataclass
class PretrainRVHeadConfig:
    """RV prediction head configuration."""
    hidden_dim: int = 256
    num_layers: int = 2
    dropout: float = 0.1


@dataclass
class PretrainTrainConfig:
    """Training configuration."""
    # Batch and accumulation
    batch_size: int = 4
    grad_accum_steps: int = 8
    effective_batch_size: int = 32  # batch_size * grad_accum_steps
    
    # Mixed precision
    amp: bool = True
    amp_dtype: str = 'bfloat16'
    
    # Training duration
    epochs: int = 100
    steps_per_epoch: int = 500
    val_steps: int = 100
    
    # Optimizer
    optimizer: str = 'adamw'
    lr: float = 1e-4
    weight_decay: float = 1e-5
    betas: tuple = (0.9, 0.999)
    
    # LR Schedule
    scheduler: str = 'cosine'
    warmup_epochs: int = 5
    min_lr: float = 1e-6
    
    # Gradient clipping
    clip_grad_norm: float = 1.0
    
    # Early stopping
    early_stopping_patience: int = 20
    
    # Loss
    loss_type: str = 'huber'  # 'mse', 'huber', 'l1'
    huber_delta: float = 0.1
    
    # Checkpointing (relative to PROJECT_ROOT)
    checkpoint_dir: str = str(PROJECT_ROOT / 'checkpoints' / 'tcn_pretrain')
    save_every_epochs: int = 10
    
    # Logging
    log_every: int = 1  # Log every batch for full visibility
    wandb_project: Optional[str] = None  # Set to enable W&B logging
    wandb_run_name: Optional[str] = None
    
    # Reproducibility
    seed: int = 42
    
    # Device
    device: str = 'cuda'
    num_workers: int = 4


@dataclass 
class PretrainConfig:
    """Complete pretraining configuration."""
    data: PretrainDataConfig = field(default_factory=PretrainDataConfig)
    tcn: PretrainTCNConfig = field(default_factory=PretrainTCNConfig)
    rv_head: PretrainRVHeadConfig = field(default_factory=PretrainRVHeadConfig)
    train: PretrainTrainConfig = field(default_factory=PretrainTrainConfig)
    
    def __post_init__(self):
        """Ensure consistency between configs."""
        # TCN input dim should match data dim
        self.tcn.dim_in = self.data.dim_in


def get_pretrain_config(**overrides) -> PretrainConfig:
    """
    Get pretraining config with optional overrides.
    
    Example:
        config = get_pretrain_config(
            data={'spy_data_path': '/path/to/spy'},
            train={'epochs': 50, 'lr': 3e-4}
        )
    """
    config = PretrainConfig()
    
    for section, values in overrides.items():
        if hasattr(config, section) and isinstance(values, dict):
            section_config = getattr(config, section)
            for key, value in values.items():
                if hasattr(section_config, key):
                    setattr(section_config, key, value)
    
    return config
