export PYTHONPATH=./
python src/cli_train.py \
--model_name VietAI/vit5-large \
--output_dir /HDD/hungnv/crag \
--data_directory /home/bakerdn/CRAG/data/dataset \
--label_file_path /home/bakerdn/CRAG/data/labels.json
