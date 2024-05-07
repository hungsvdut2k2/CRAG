from dataclasses import dataclass, field


@dataclass
class ModelArguments:
    model_name: str = field(
        default=200,
        metadata={"help": "Model name or path."},
    )
    num_labels: int = field(default=2, metadata={"help": "The number of label in training dataset"})
