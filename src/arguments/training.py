from dataclasses import dataclass, field


@dataclass
class TrainingProcessArguments:
    seed: int = field(default=42, metadata={"help": "random seed for training"})
    num_epochs: int = field(
        default=200,
        metadata={"help": "Number of training epochs"},
    )
    label_smoothing_factor: float = field(
        default=0.01,
        metadata={"help": "Label smoothing factor value"},
    )
    output_dir: str = field(default="", metadata={"help": "Directory for saving model"})
    learning_rate: float = field(
        default=2e-5, metadata={"help": "Learning rate for training process"}
    )
    weight_decay: float = field(
        default=0.01, metadata={"help": "Weight Decay for training process"}
    )
    warmup_steps: int = field(
        default=500, metadata={"help": "Warmup steps for training process"}
    )
    batch_size: int = field(
        default=8, metadata={"help": "Batch size for training process"}
    )
    gradient_accumulation_steps: int = field(
        default=1, metadata={"help": "Gradient Acummulation Steps for training process"}
    )
    label_smoothing_factor: float = field(
        default=0.1, metadata={"help": "Label smoothing for training process"}
    )
    trainer_type: str = field(
        default="base", metadata={"help": "Trainer type for training process"}
    )
    gamma: int = field(
        default=2, metadata={"help": "Gamma hyperparameter for training process"}
    )
    alpha_1: int = field(
        default=3,
        metadata={"help": "Alpha 1 hyperparamter for training process"},
    )
    alpha_2: int = field(
        default=5, metadata={"help": "Alpha 2 hyperparameter for training process"}
    )
    device: str = field(default="cuda", metadata={"help": "Device for trainig"})
