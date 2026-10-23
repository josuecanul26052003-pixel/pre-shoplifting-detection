import torchmetrics
from torchmetrics.classification.precision_recall import BinaryRecall, BinaryPrecision
from torchmetrics.classification.accuracy import BinaryAccuracy
from torchmetrics.classification.f_beta import BinaryF1Score
from torchmetrics.classification.auroc import BinaryAUROC
from torchmetrics.classification.average_precision import BinaryAveragePrecision
import numpy as np

METRICS_TO_PLOT = ['accuracy', 'loss', 'f1']

METRICS = torchmetrics.MetricCollection(
            {
                "accuracy":BinaryAccuracy(),
                "f1": BinaryF1Score(),
                "precision": BinaryPrecision(),
                "recall": BinaryRecall(),
                "auc_roc": BinaryAUROC(),
                "auc_pr": BinaryAveragePrecision()
            }
        )

METRICS_MULTI = torchmetrics.MetricCollection(
            {
                "accuracy":torchmetrics.Accuracy(task='multiclass', num_classes=2),
                "f1": torchmetrics.F1Score(task='multiclass',average='macro', num_classes=2),
                "precision": torchmetrics.Precision(task='multiclass',average='macro', num_classes=2),
                "recall": torchmetrics.Recall(task='multiclass',average='macro', num_classes=2),
                "auc_roc": torchmetrics.AUROC(task='multiclass',average='macro', num_classes=2),
                "auc_pr": torchmetrics.AveragePrecision(task='multiclass',average='macro', num_classes=2)
            }
)