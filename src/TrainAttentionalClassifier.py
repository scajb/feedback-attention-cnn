# DO NOT REMOVE PANDAS IMPORT - CAUSES "version `GLIBCXX_3.4.21' not found" ERROR
# noinspection PyUnresolvedReferences
import pandas as pd
# Uncomment for extra debugging around gradient backpropagation
# import torch

from src.request.TrainClassifierRequest import TrainClassifierRequest
from device import get_device
from logging_support import init_logging, log_info
from src.trainer.AttentionCNNTrainerFactory import AttentionCNNTrainerFactory

if __name__ == "__main__":
    """
    TrainAttentionalClassifier.py
    
    Trains Attention-based CNN to classify histolpathological image patches, according to class names represented 
    by subdirectories of image store.
    
    Args:
    
    0) Path to this script
    1) Path to project root directory
    2) Number of training epochs
    3) Path where trained model will be saved
    4) Path to log output file
    5) Optional path to existing model, when resuming training from earlier session
    6) Optional experiment ID, for looking up test/training/validation split
    7) SQLite database path, where experiment IDs are stored
    8) 
    """

    request = TrainClassifierRequest.parse_args()
    init_logging(request.log_path)
    log_info(request)

    trainer = AttentionCNNTrainerFactory.get_attention_cnn_trainer(request, get_device(request.use_cpu))

    # Uncomment for extra debugging around gradient backpropagation
    # torch.autograd.set_detect_anomaly(True)

    trainer.train_classifier(request)
