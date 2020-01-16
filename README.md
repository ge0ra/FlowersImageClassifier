# Flowers Image Classifier with Deep Learning

Application built with PyTorch during the [Udacity's AI Programming with Python Nanodegree program](https://eu.udacity.com/course/ai-programming-python-nanodegree--nd089).

The application currently has training, validation, and test data for 102 species of flowers and uses transfer learning with either Densenet121 or VGG13 to train and infer with.

## Usage
The following instructions assume that you:
1. Have git installed on your machine. If not, you can click [here](https://git-scm.com/book/en/v2/Getting-Started-Installing-Git) to set it up.
2. Have a package manager installed. The following instructions use [conda](https://docs.conda.io/en/latest/) but [pip](https://pypi.org/project/pip/) works just as well.


First, clone the repository with git:

`git clone https://github.com/ge0ra/image-classifier-deep-learning.git`

Then, open your terminal and change the current working directory into the repository

`cd ../image-classifier-deep-learning`

Install the following packages:
1. Python 3.7: `conda install python==3.7`
2. Numpy 1.16: `conda install numpy==1.16`
3. Pytorch 1.1: `conda install pytorch=1.1`
4. Torchvision 0.3: `conda install torchvision=0.3`
5. Matplotlib 3.0: `conda install matplotlib==3.0`


## How to use
The first step is to train a model on the training and validation data available in the "flowers" folder. You can then save a checkpoint and load it again to make predictions with the test data folder.

### Training a model
Once you are in the working directory of the repository to train a model (Densenet121 or VGG13) you can run the `train.py` file like so:

`python train.py flowers`

The `flowers` file path is the data directory that includes the train and valid folders that contain the training and validation images. By default the Densenet121 model will be used along with other pre-set hyper-parameters. Here is a list of all of the arguments you can use in training along with some examples:

1. data_directory:
- The relative path to the image folder to train on. Two folders are mandatory for the training: 'train' and 'valid'.
2. --save_dir:
- The relative path to the directory you wish to save the trained model's checkpoint to. This file path must exist prior to training
- Default is the checkpoint directory 'checkpoints'
3. --arch:
- The architecture you would like to use for the training. This can either be 'densenet121' or 'vgg13'
- Default is densenet121
4. --learning_rate:
- The learning rate for the training process
- Default is 0.003
5. --hidden_units:
- The number of units used in the hidden layer. NOTE: There is only one hidden layer used in this project and thus only one hidden_unit required for training
- Default is 500
6. --epochs:
- The amount of epochs to train for.
- Default is 3
7. --gpu:
- An option if you would like to use the GPU for training.

Example use of all arguments:

`python train.py flowers --save_dir checkpoints --arch densenet121 --learning_rate 0.03 --hidden_units 500 --epochs 2 --gpu`


### Inference with the trained model
Now that the model is trained, you can use it to infer the species of a flower on the pictures of the `test` folder. To do so use the `predict.py` file like that:

`python predict.py flowers/test/1/image_06743.jpg checkpoints/checkpoint.pth`

The `flowers/test/23/image_03382.jpg` is the file path and name of the image we wish to infer on. The file extension is required. By default this will return the top 1 prediction. Here is a list of arguments you can use to get the top n predictions or use the GPU for inference

1. data_directory:
- The relative path to the image file that you want to infer on. The file name and extension are required.
2. checkpoint:
- The relative path to the models checkpoint pth file. The file name and extension are required.
3. --top_k:
- The amount of most likely classes to return for the predictions
- Default is 1
4. --category_names:
- The json file, including file path, to load category names
- Default is 'cat_to_name.json'
5. --gpu:
- An option if you would like to use the GPU for training.



## Licence
[MIT](https://opensource.org/licenses/MIT)
