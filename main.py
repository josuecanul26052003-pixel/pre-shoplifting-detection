import lightning as L
import os
from dotenv import load_dotenv
from src.modules.datamodule import SuspiciousDataModule
from src.utils import load_database, plot_metric_from_csv, plot_confusion_matrix, compute_stadistics
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import ModelCheckpoint
from ncps.torch import LTC
from ncps.wirings import AutoNCP
from src.modules.model import BaseModel
from src.consts import METRICS_MULTI, METRICS_TO_PLOT
import torch.nn.functional as F
load_dotenv('.config')

N_FOLDS = int(os.getenv("N_FOLDS"))
DATASET = load_database('dataset/final/features/dataset.pkl')
BATCH_SIZE = int(os.getenv("BATCH_SIZE"))
SEEDS = [42, 47, 50]
SAVE_DIR = os.getenv("SAVE_DIR")
MAX_EPOCHS = int(os.getenv('MAX_EPOCHS'))
LR = float(os.getenv("LR"))
for seed in SEEDS:
    
    for fold in range(N_FOLDS):
        # Config
        data_path = f'data/k_fold/{fold}'
        
        suspicious_data_module = SuspiciousDataModule(DATASET, k=fold, seed=seed, batch_size=BATCH_SIZE)
        suspicious_data_module.prepare_data()
        suspicious_data_module.setup()
        csv_logger = CSVLogger(save_dir=SAVE_DIR, name='', version=f'seed_{seed}_fold_{fold}')

        logger_dir = csv_logger.log_dir
        
        checkpoint_callback = ModelCheckpoint( 
            monitor='val_f1', 
            dirpath=logger_dir, 
            filename='best', 
            save_top_k=1, 
            mode='max'
        )

        # Model
        OUT_FEATURES = 2
        IN_FEATURES = 4132
        wiring = AutoNCP(8, OUT_FEATURES)
        ltc_model = LTC(IN_FEATURES, wiring, batch_first=True)
        lit_model = BaseModel(METRICS_MULTI, ltc_model,  F.cross_entropy, learning_rate=LR, task='multiclass', logits_convertion=lambda x:x)
        trainer = L.Trainer(
            max_epochs=MAX_EPOCHS,
            logger=csv_logger,
            callbacks=[checkpoint_callback]
        )
        trainer.fit(lit_model, suspicious_data_module)
        trainer.test(datamodule=suspicious_data_module, ckpt_path='best')

        metric_csv_path = os.path.join(logger_dir, 'metrics.csv')
        if os.path.exists(metric_csv_path):
            for metric in METRICS_TO_PLOT:
                plot_metric_from_csv(metric_csv_path, metric)
plot_confusion_matrix(SAVE_DIR)
compute_stadistics(SAVE_DIR)
