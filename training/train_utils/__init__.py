"""Models and loss functions module"""
from .physics_informed_mae import PhysicsInformedMAE, create_physics_informed_mae
from .fisr import FISR, FISRx4, create_fisr_x2, create_fisr_x4
from .modules import (
    PhysicsInformedPreprocessing,
    MaskedPatchEmbedding,
    FluxGuidanceGeneration,
    FluxGuidanceController,
    TransformerBlock,
)
from .losses_sr import (
    CharbonnierLoss,
    SSIMLoss,
    FFTLoss,
    MaskedReconstructionLoss,
    compute_multiresolution_losses,
)

__all__ = [
    'PhysicsInformedMAE',
    'create_physics_informed_mae',
    'FISR',
    'FISRx4',
    'create_fisr_x2',
    'create_fisr_x4',
    'PhysicsInformedPreprocessing',
    'MaskedPatchEmbedding',
    'FluxGuidanceGeneration',
    'FluxGuidanceController',
    'TransformerBlock',
    'CharbonnierLoss',
    'SSIMLoss',
    'FFTLoss',
    'MaskedReconstructionLoss',
    'compute_multiresolution_losses',
]
