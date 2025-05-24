# A Model Extraction Attack on Deep Neural Networks Running on GPUs
Code for the theiss: [A Model Extraction Attack on Deep Neural Networks Running on GPUs](https://doi.org/10.7275/35614344)

By [Jonah O'Brien Weiss](https://jonahobw.github.io)

This repo contains code automating large scale experiments to:
- Train deep neural networks (DNNs) in Pytorch
- Profile DNNs using NVIDIA's `nvprof` tool and parse them
- Train architecture prediction models on datasets of profiles
- Train surrogate DNNs in a black-box setting
- Run model evasion attacks on victim DNNs and transfer attacks from the surrogate to victim DNNs
- Complete a full model extraction attack recovering the architecture and approximate weights of the victim model
- Perform comprehensive data collection, metrics calculation, and plotting

## Installation and Requirements

Software Requirements:
* Linux OS
* Python 3.9
* CUDA at least v10.2 with nvprof installed (use ```nvprof --help``` to check)

Hardware Requirements:
* Nvidia GPU.  This was tested with the Quadro GTX 8000 and Tesla T4 GPUs.

Install requirements with 
```
pip3 install -r requirements.txt
```

## Directory Structure

Notable files are outlined below.

```yaml
src/:
  architecture_prediction.py  # training a model on a set of profiles
  collect_profiles.py         # automate profile collection
  construct_input.py          # the input to DNN when profiling
  create_exe.py               # create an executable for profiling
  data_engineering.py         # validating and massaging data
  datasets.py                 # manages image classification datasets
  download_dataset.py         # downloads a dataset from datasets.py
  email_sender.py             # configure email notification for long-running experiments
  experiments.py              # experiment management and execution
  format_profiles.py          # parser for nvprof output
  get_model.py                # wrapper for getting DNN models
  logger.py                   # utility for logging during training
  model_inference.py          # target file for the executable, this is what will be profiled
  model_manager.py            # training, profiling, and attacking victim DNNs
  model_metrics.py            # accuracy calculations
  neural_network.py           # custom model for architecture prediction
  online.py                   # calculate stats while training DNNs
  utils.py                    # general utility functions
  whitebox_pyprof.py          # using PyTorch's builtin profiler

  plots/:                     # code for generating plots

  profiles/:                  # profiling data and utilities
    debug_profiles/           # example profiles

  tensorflow/:                # tensorflow-specific code
    create_exe.py             # creates an exe for tensorflow profiling
    tensorflow_inference.py   # target file for tensorflow executables

  test/                       # test code

  notebooks/                  # Jupyter notebooks, runnable on Google Colab
```

## Collecting a Dataset of Profiles
### Quick Start: Use Google Colab
1. Go to [Google Colab](https://colab.research.google.com/).  Note that you will need to add a GPU, [this example](https://colab.research.google.com/notebooks/gpu.ipynb) demonstrates how.
2. Run the notebook [ProfileVictimModels.ipynb](src/notebooks/ProfileVictimModels.ipynb).

### To run on your own GPU:
1. Create an executable for your device. Run ```python create_exe.py```
2. Collect profiles.  Run ```python collect_profiles.py```
3. Validate and parse profiles.  Run ```python format_profiles.py```.  This will validate profiles and class balance and ask you for permission to remove extra or faulty profiles.  If an error occurs, you will need to run this again.

## Train an Architecture Prediction Model

The architecture prediction model maps a profile to the architecture of the DNN that generated the profile.

The following code shows how to train an architecture prediction model:

```python
from data_engineering import all_data
from architecture_prediction import get_arch_pred_model

data = all_data("src/profiles/debug_profiles")
arch_pred_model = get_arch_pred_model("rf", df=data)  # "rf" = random forest, "lr" = linear regression, "nn" = neural net ...
print(arch_pred_model.evaluateTest())
print(arch_pred_model.evaluateTrain())
```
