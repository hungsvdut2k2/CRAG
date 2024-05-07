def preprocess_function(examples, tokenizer):
    batch_encoding = tokenizer(examples["text"], padding=True, truncation=True, max_length=1024)
    return batch_encoding
