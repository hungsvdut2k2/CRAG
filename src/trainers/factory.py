from transformers import Trainer

from .focal_loss_trainer import FocalLossTrainer


def get_trainer(trainer_type: str, **kwargs) -> Trainer:
    if trainer_type == "base":
        return Trainer(**kwargs)
    elif trainer_type == "focal_loss":
        return FocalLossTrainer(**kwargs)
    else:
        raise NotImplementedError
