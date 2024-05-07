import numpy as np
from seqeval.metrics import classification_report, f1_score, precision_score, recall_score


def compute_metrics(eval_pred, id2label):
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    y_true = labels.tolist()
    y_pred = predictions.tolist()

    y_true = [id2label[label] for label in y_true]
    y_pred = [id2label[label] for label in y_pred]

    y_true = [["B-" + str(label)] for label in y_true]
    y_pred = [["B-" + str(label)] for label in y_pred]
    metrics = classification_report(y_true=y_true, y_pred=y_pred, digits=3)
    print(metrics)

    precision = precision_score(y_true=y_true, y_pred=y_pred, average="macro")
    recall = recall_score(y_true=y_true, y_pred=y_pred, average="macro")
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average="macro")

    return {"precision": precision, "recall": recall, "f1-score": f1}
