"""FISR training configuration."""

from dataclasses import dataclass


@dataclass
class FISRTrainingConfig:
    # Model
    variant: str = "x2"  # x2 or x4
    inp_channels: int = 1
    out_channels: int = 1
    dim: int = 48
    num_blocks: tuple[int, int, int, int] = (4, 6, 6, 8)
    num_refinement_blocks: int = 4
    heads: tuple[int, int, int, int] = (1, 2, 4, 8)
    ffn_expansion_factor: float = 2.66
    bias: bool = True
    layer_norm_type: str = "WithBias"
    decoder: bool = True
    use_loss: str = "L1"
    use_attention: bool = False

    # Train
    batch_size: int = 8
    num_epochs: int = 100
    learning_rate: float = 2e-4
    weight_decay: float = 1e-5

    # Data
    data_dir: str = "dataset/data/x2"
    num_workers: int = 4

    # Runtime
    device: str = "cuda"
    seed: int = 42

    # Logging/checkpoints
    save_dir: str = "checkpoints/fisr"
    save_interval: int = 5
    log_interval: int = 10
    save_every_batches: int = 0

    def to_dict(self) -> dict:
        return self.__dict__

    @classmethod
    def from_dict(cls, config_dict: dict) -> "FISRTrainingConfig":
        return cls(**config_dict)
