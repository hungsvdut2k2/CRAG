from dataclasses import dataclass, field


@dataclass
class DatasetArguments:
    data_directory: str = field(
        default="",
        metadata={"help": "The Directory Of Dataset"},
    )

    label_file_path: str = field(default="", metadata={"help": "The file path of label file"})
    num_process: int = field(
        default=1,
        metadata={"help": "The number of process used for preprocessing dataset"},
    )
