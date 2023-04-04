import torch.cuda

from logging_support import log_info


# noinspection PyUnresolvedReferences
def get_device(use_cpu=False):
    gpu = torch.cuda.is_available() and not use_cpu
    _device = torch.device('cuda:0' if gpu else 'cpu')
    log_info("Device:", _device)
    if gpu:
        log_info("Torch CUDA available:", torch.cuda.is_available())
        log_info("Torch version:", torch.__version__)
        log_info("Torch CUDA version:", torch.version.cuda)
        log_info("Torch backends cudnn version:", torch.backends.cudnn.version())
    return _device
