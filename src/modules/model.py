from abc import ABC
import torchmetrics
import torch
import lightning as L
from torchmetrics.classification import BinaryConfusionMatrix, ConfusionMatrix
import torch.nn.functional as F

class BaseModel(L.LightningModule):
    def __init__(self, metrics, 
                 model: torch.nn.Module,
                 criteria,
                 learning_rate=1e-3,
                 task: str = 'multiclass',
                 logits_convertion = lambda x: x
                 ):
        super().__init__() # ¡Súper importante no olvidar esto!
        self.learning_rate = learning_rate
        self.model = model
        self.train_tracker = torchmetrics.wrappers.MetricTracker(metrics.clone(prefix='train_'))
        self.val_tracker = torchmetrics.wrappers.MetricTracker(metrics.clone(prefix='val_'))
        self.test_tracker = torchmetrics.wrappers.MetricTracker(metrics.clone(prefix='test_'))
        self.test_cm = BinaryConfusionMatrix() if task == 'binary' else ConfusionMatrix(task='multiclass', num_classes=2)
        self.criteria = criteria
        self.task = task
        self.logits_convertion = logits_convertion
            
    def forward(self, batch):
        # 1. Extraer características y etiquetas del diccionario
        x = batch['features']
        labels = batch['label'].long()  # FIX 1: float32 → long requerido por cross_entropy
        
        # 2. El modelo LTC devuelve una tupla (salida, estado_oculto). Ignoramos el estado oculto con '_'
        logits, _ = self.model(x)
        
        # 3. Extraer el último paso temporal para quitar la dimensión de secuencia.
        # Transforma logits de [Batch, Secuencia, Clases] (ej. [8, 1, 2]) a [Batch, Clases] (ej. [8, 2])
        logits = logits[:, -1, :] 
        
        # 4. Si tienes alguna conversión extra definida, la aplicas aquí
        if hasattr(self, 'logits_convertion'):
            logits = self.logits_convertion(logits)  # shape: [B, num_classes]
            
        # 5. Calcular la pérdida
        loss = self.criteria(logits, labels)  # FIX 2: (predicción, target) en orden correcto
        
        return loss, logits, labels

    def training_step(self, batch, batch_idx):
        loss, logits, y = self(batch)
        self.train_tracker.update(logits, y)  # FIX 3: logits [B,C] y labels [B], sin one_hot
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, logits, y = self(batch)
        self.val_tracker.update(logits, y)  # FIX 3
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch, batch_idx):
        loss, logits, y = self(batch)
        self.test_tracker.update(logits, y)  # FIX 3
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.test_cm.update(logits.argmax(dim=1), y)  # FIX 4: CM espera predicciones discretas

    def on_train_epoch_start(self):
        self.train_tracker.increment()
        
    def on_validation_epoch_start(self):
        self.val_tracker.increment()
        
    def on_test_epoch_start(self):
        self.test_tracker.increment()

    def on_train_epoch_end(self):
        self.log_dict(self.train_tracker.compute())

    def on_validation_epoch_end(self):
        self.log_dict(self.val_tracker.compute())
        
    def on_test_epoch_end(self):
        self.log_dict(self.test_tracker.compute())
        cm = self.test_cm.compute()
        self.log("test_TN", cm[0, 0])
        self.log("test_FP", cm[0, 1])
        self.log("test_FN", cm[1, 0])
        self.log("test_TP", cm[1, 1])
        self.test_cm.reset()
        
    def configure_optimizers(self):
        # return torch.optim.AdamW(self.parameters(), lr=self.learning_rate)
        return torch.optim.NAdam(self.parameters(), lr=self.learning_rate)
        # return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        