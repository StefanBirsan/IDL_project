"""
Metric tracking for training
"""
from typing import Dict, List
from dataclasses import field, dataclass


@dataclass
class MetricTracker:
    """Track training and evaluation metrics"""
    train_losses: List[Dict[str, float]] = field(default_factory=list)
    eval_losses: List[Dict[str, float]] = field(default_factory=list)
    learning_rates: List[float] = field(default_factory=list)
    
    def add_train_loss(self, loss_dict: Dict[str, float]):
        """Add training loss"""
        self.train_losses.append(loss_dict)
    
    def add_eval_loss(self, loss_dict: Dict[str, float]):
        """Add evaluation loss"""
        self.eval_losses.append(loss_dict)
    
    def add_learning_rate(self, lr: float):
        """Add current learning rate"""
        self.learning_rates.append(lr)
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'train_losses': self.train_losses,
            'eval_losses': self.eval_losses,
            'learning_rates': self.learning_rates,
        }
    
    def get_best_eval_loss(self) -> float:
        """Get best evaluation loss"""
        if not self.eval_losses:
            return float('inf')
        return min([loss['loss_total'] for loss in self.eval_losses])
