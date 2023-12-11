# feedback-attention-cnn
This repository contains implementations of CNN classifier models using Feedback Attention and Saccade behaviour,
introduced in our paper _Object-based Attention Improves Tumour Detection in Digital Pathology_. 

A link to the paper will be supplied once a preprint edition has been published online. 

## Feedback Attention Ladder CNN

The script _**ExecuteFeedbackAttentionCNN.py**_ instantiates a Feedback Attention Ladder CNN (FAL-CNN) feedback attention model,
initialises its weights from a specified file then executes the FAL-CNN against a specified input image. 

The feedback attention model outputs a predicted class, which is reported to a specified log file. 

The model also returns a collection of its feedback activations, at different levels in the model. 
These are used to generate plots to visualise the model's attention regions, 
superimposed on the original input image.

For ImageNet-100 input images, further plots are generated showing the region of the XML bounding box 
annotation where available.

### Function arguments

The **_ExecuteFeedbackAttentionCNN_** script requires the following inputs. 
If running in PyCharm, these can be configured in the Run Configuration 
based on the _ExecuteFeedbackAttentionCNN_ example supplied. 

1. Path to attention weights of pre-trained FAL-CNN model, as .PTH file.
2. Path to RGB input image, e.g. in ImageNet-100 Test set, as JPEG or PNG
3. Path for local log file output
4. Path to output directory, where generated images will be saved
5. Optional path to directory of XML files containing ImageNet bounding box annotations

## Saccade model

The script _**ExecuteSaccadeModel.py**_ constructs and executes a Saccade model,
using an embedded FAL-CNN feedback attention model to sample a sequence of  
224x224px image regions within a 448x448px input image. The inner region is 
initially taken from the centre of the larger image, then tracks to follow the 
Centre of Attention (CoA) derived from mean feedback activations in the FAL-CNN.

This script outputs visualisation plots showing sampled regions with overlaid 80%
attention contours and their centroids, for the initial region and each 
subsequent saccade iteration.

### Function arguments

The _**ExecuteSaccadeModel**_ script requires the following inputs. 
If running in PyCharm, these can be configured in a Run Configuration 
based on the _ExecuteSaccadeModel_ example supplied.

1. Path to attention weights of pre-trained FAL-CNN model, as .PTH file.
2. Path to RGB input image, e.g. in ImageNet-100 Test set, as JPEG or PNG
3. Path for local log file output
4. Output directory path for feedback visualisation plots
5. Number of saccade iterations required

## Saccade model: Jupyter notebook for composite plots

Once _ExecuteSaccadeModel_ has been executed for an input image, the 
Jupyter notebook _**SaccadeViewer.ipynb**_ can be used to generate a composite plot
showing the sequence of saccade movements and associated regions of high attention.

![Screenshot](saccade-sequence-example-1.png)

Local variables **_input_dir_** and **_saccade_inputs_** can be configured according to 
file names and directory locations used in _ExecuteSaccadeModel_.
## Data requirements (data supplied separately from this repo)

1. Pre-trained feedback model weights, matching FeedbackAttentionLadderCNN class in _classes/classifier_.
2. ImageNet-100 Test set, downloaded to local directory
3. Optional ImageNet bounding box annotations (XML files)

## Python environment

The following shell commands are used to create and configure a Python environment supporting execution
on an HPC environment or local Anaconda installation. 
It is recommended to run these commands against a new, named Conda environment for this project, 
to protect pre-existing package versions in your base environment. 

Note that the PyTorch versions were chosen for compatibility with the CUDA drivers on the HPC nodes used by the author. 
If your local installation uses a different version of CUDA you may need to substitute compatible PyTorch versions, 
per https://discuss.pytorch.org/t/pytorch-for-cuda-10-2/65524 and https://pytorch.org/get-started/previous-versions/.
If no CUDA installation exists, the above code will automatically run on the available CPU instead.
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

