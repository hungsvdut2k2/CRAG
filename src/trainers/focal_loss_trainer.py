from src.losses.focal_loss import FocalLoss
from transformers import Trainer


class FocalLossTrainer(Trainer):
    def __init__(
        self,
        *args,
        alpha,
        gamma,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.alpha = alpha
        self.loss = FocalLoss(gamma=gamma, alpha=alpha)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs[0]
        loss = self.loss(logits, labels)
        return (loss, outputs) if return_outputs else loss
