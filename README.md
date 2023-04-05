# feedback-attention-cnn
CNN classifier models using Feedback Attention

## ExecuteFeedbackAttentionCNN

This script loads a feedback attention model from file, then executes it against a specified input image. 

If bounding box XML annotations are available, these are also loaded.

The feedback attention model outputs a predicted class, which is reported to a log file. 

The model also returns a collection of its feedback activations, at different levels in the model. These are used to 
generate plots to visualise the model's attention regions, against the original input image.

Further plots are generated showing the region of the XML bounding box, if available.

### Function arguments

The ExecuteFeedbackCNN functoin requires the following inputs. If running in PyCharm, these can be configured in the ExecuteFeedbackAttentionCNN run configuration. 

1. Path to pre-trained feedback attention model, as .PT file. Should be of one of the the PyTorch CNN implementations in the classes/classifier subdirectory of this project.
2. Path to input image, e.g. from ImageNet-100 test set, as JPEG or PNG
3. Path to local log file output
4. Path to output directory, where generated images will be saved
5. Path to directory of XML files containing bounding box annotations

### Data requirements (data supplied separately from this repo)

1. Pre-trained feedback model(s), matching torch.nn.Module-derived classes in classes/classifier
2. ImageNet-100 test set, with subdirectories renamed as 'xxx_class_name' for readability
3. Bounding box annotations (XML files)

### Python environment

The following shell commands have been used successfully when executing similar code on an HPC environment. It is recommended to run these against a new, named Conda environment for this project, to avoid breaking existing packages in your local base environement. 

Note that the PyTorch versions were chosen for compatibility with the CUDA drivers on the ARC4 HPC nodes. If your local installation uses a different version of CUDA you may need to select different versions. See
https://discuss.pytorch.org/t/pytorch-for-cuda-10-2/65524 and https://pytorch.org/get-started/previous-versions/.
```
conda create --name feedback-attention-env python=3.7
source activate feedback-attention-env

conda install setuptools=45.2.0
pip install libarchive openslide-python
conda install pandas

pip3 install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html
pip install sklearn tqdm matplotlib scikit-image shapely descartes efficientnet_pytorch python-interface opencv-python
pip install -U jsonpickle
```

