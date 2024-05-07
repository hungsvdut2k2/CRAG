from transformers import HfArgumentParser
from arguments import DatasetArguments, ModelArguments, TrainingProcessArguments
from pipelines import TrainingPipeline


if __name__ == "__main__":
    parser = HfArgumentParser(
        (DatasetArguments, ModelArguments, TrainingProcessArguments)
    )
    data_args, model_args, train_args = parser.parse_args_into_dataclasses()

    training_pipeline = TrainingPipeline(
        dataset_arguments=data_args,
        model_arguments=model_args,
        training_arguments=train_args,
    )

    training_pipeline.train()
