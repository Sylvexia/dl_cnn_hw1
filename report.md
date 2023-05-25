# 深度學習 作業一
A1085125 洪祐鈞(Sylvex Hung) 2023/02/24

## Disclaimer

- The code in this homework is amalgamation of LLM prompt (ChatGPT and discord Clyde), GitHub Copilot autocompletion, and random code from the forum. I would provide the prompt of the LLM to llm_prompt.md, also keep the comment for the code generated from copilot.
- I would also list some reference during my research, for the transparancy.

## Environments

- O.S: ManjaroLinux 22.1.2 Talos
- Miniconda 23.1.1
- ASUS aspire-7 (A715-51G) laptop.
- CPU: intel i7-1260p
- GPU: NVIDIA GeForce RTX 3050 4GB Laptop GPU
- Python 3.11.3
- For python module version, please refer to requirements.txt

## How to run?

Referring requirements.txt to set up your environment, preferably use a conda first so it would not messed up your local environment.

Running: ```python ex_1.py``` for first experiment, there are 5 experiments for respective experiments, it would start training and testing cifar10/cifar100 dataset for a model. Specifically, like the following:
```
if __name__ == '__main__':
    train_cifar_10()
    train_cifar_100()
```
After running any experiments, a folder named ```ex_{number}``` would be generated, which contains two folder ```cifar_10``` and ```cifar_100```, inside the folers, it would generate time-stampped folder for training/testing a single model, inside the timestampped folder, contains the important files as follows:

- train.json: training log
- test.json: test log
- loss.png: loss per epoch during training
- accuracy.png: train/validation accuracy during training
- confusion_matrix.png: as the name shows.
- best.pth: the best model best validation accuracy during training.
- epoch-{num}: model file for checkpoint.

The log contains training/testing complete time, accuracy/loss per epoch during training, test metrics(accuracy, ioc score, precision, F1_score, and etc.) I would do a comparison at comparision, if you want to see more fine-grained version, you can check that out!
(tips: use json formater can help you see the log if your ide has one.)

Also it would also generate data for cache-ing the cifar10/cifar100 datasets.

## General Strategy of the homework

- The computational resources is limited, instead of choosing a bulky model which has highest accuracy. What I care about is the efficiency of the model, which means I can get high accuracy in short amount of time. This way, I can have more feedback 
- Running on local computer, which I can modularize my code and having better developter experience. ~~Also, I love copilot.~~
- Keeping the training/testing log, and visualize the result. For better knowing what exactly happened during training and testing.
- Crammming as much of the batch size as my GPU can handle.

## Experiments

### EX-1

Basically, the model is the same as what the professor gave us. What's different is the batch size is 128, and the number of epoch is 50.

The model architecture is as follows: (num_classes = 100 for cifar100)

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 12, 30, 30]             912
       BatchNorm2d-2           [-1, 12, 30, 30]              24
            Conv2d-3           [-1, 12, 28, 28]           3,612
       BatchNorm2d-4           [-1, 12, 28, 28]              24
         MaxPool2d-5           [-1, 12, 14, 14]               0
            Conv2d-6           [-1, 24, 12, 12]           7,224
       BatchNorm2d-7           [-1, 24, 12, 12]              48
            Conv2d-8           [-1, 24, 10, 10]          14,424
       BatchNorm2d-9           [-1, 24, 10, 10]              48
           Linear-10                  [-1, 100]         240,100
================================================================
Total params: 266,416
Trainable params: 266,416
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 0.42
Params size (MB): 1.02
Estimated Total Size (MB): 1.44
----------------------------------------------------------------
```

Here's Cifar 10 results:

![picture 1](images/f7e0e16dc1ab278656134d68fd2effb041c36c9fce0edb3edde1b2db41172b37.png)

![picture 3](images/9f53b04cf881a022f7cacc68c229290a2a6cabf480491171124c745e546b7d0e.png)

```
# Training:
"best_val_accuracy": 71.94%,
"best_epoch": 13,
"train_time": 124.0 seconds
```

```
# Testing:
"accuracy": 72.24%,
"losses": 0.8672,
"testing time": 0.4442 seconds,
"precision_score": 0.7325,
"recall_score": 0.7224,
"f1_score": 0.7255,
"auc_score": 0.8458,
```

Here's Cifar 100 results:

![picture 2](images/7ef38fea291b1cc8ffd9f9db29f19d7ba64136c19b7ea1f8468396e13ecfe4c7.png)

![picture 4](images/d6170958a214e01c99cd268ab2368f73a7410bf38dd5d8d45f70dcb0f24ea595.png)

```
# Training: 
"best_val_accuracy": 35.25%,
"best_epoch": 6,
"train_time": 127.2 seconds
```

```
# Testing
"accuracy": 34.65%, 
"losses": 2.6727387393951414,
"testing time": 0.486 seconds,
"precision_score": 0.374,
"recall_score": 0.3465,
"f1_score": 0.3389,
"auc_score": 0.6699,
```

As the result shows, it have overfitting problem. Especially cifar100. Since the validation loss goes sky-rocket. My speculation of why cifar100 has massive overfitting may due to:

1. The image number per class is small, which provide the same information over and over again.
2. The image size is too small, provide little information for a model to learn.
3. The model is too small, which can not extract too much feature.

In this experiment, this is the first time I learned that what is overfitting, and longer training epoch is not the fix.

Also, fun fact: I literally ask LLM what metrics should I include in the report, and the metrics above is what it gave me, and I learned what these new metrics means on the fly. Due to the test set number of class is identical for all the classes. the recall score and the accuracy should be the same. This surprise me the first.

### EX-2

For the spculation above, I think it's good to add some changes accordingly:

1. Augmenting the number of class samples.
2. Resizing the image to larger size.
3. Making the model larger to extract more features.

Simply words: Bigger, better and stronger.

So I make the model as following:

```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1         [-1, 16, 128, 128]             448
       BatchNorm2d-2         [-1, 16, 128, 128]              32
              ReLU-3         [-1, 16, 128, 128]               0
         MaxPool2d-4           [-1, 16, 64, 64]               0
            Conv2d-5           [-1, 32, 64, 64]           4,640
       BatchNorm2d-6           [-1, 32, 64, 64]              64
              ReLU-7           [-1, 32, 64, 64]               0
         MaxPool2d-8           [-1, 32, 32, 32]               0
            Conv2d-9           [-1, 64, 32, 32]          18,496
      BatchNorm2d-10           [-1, 64, 32, 32]             128
             ReLU-11           [-1, 64, 32, 32]               0
        MaxPool2d-12           [-1, 64, 16, 16]               0
           Linear-13                 [-1, 1024]      16,778,240
             ReLU-14                 [-1, 1024]               0
           Linear-15                  [-1, 100]         102,500
================================================================
Total params: 16,904,548
Trainable params: 16,904,548
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.19
Forward/backward pass size (MB): 11.39
Params size (MB): 64.49
Estimated Total Size (MB): 76.06
----------------------------------------------------------------
```

Specifically:

- Input: ```3*128*128```
- 3 convolution layer are all ```kernel_size=3, stride=1, padding=1```
- 3 max pooling layers are all ```kernel_size=2, stride=2```.
- 2 linear layers: with 1024 and 10/100 output (cifar10/cifar100)
- Activation function are all ReLU

The reason my design was:

- In the VGG model paper, it states that using smaller size convolution can increase the receptive field.
- Also using size 3 convoltion layer is just easier to design the layer afterwards.
- Multiple linear layers seems perform better than single layer.
- With the consideration of training time, and increase the size of the image, 
I'm afraid of increasing the depth of the network.

For the data augmentation part, basically I'm refer to the artical:
[(Image Classification — Cifar100)](https://shihyung1221.medium.com/image-classification-cifar100-af751271b398): The author resize to image larger to get better result. Hence, I do the following for the augmentation:

```
transforms.Resize((128, 128), antialias=True),
transforms.RandomCrop(128, padding=16),
transforms.RandomHorizontalFlip(),
```

The way I didn't do too much augmentation has 2 reasons:
- I was trying training on WSL and kaggle, transformation cause massive bottleneck during training. The GPU did random spike with low utilization on WSL. For kaggle, it overloads the GPU. (Although I did re-train on native linux and the performance is a lot better)
- This project [kuangliu /
pytorch-cifar](https://github.com/kuangliu/pytorch-cifar/blob/master/main.py) only use the transformation above with sigle decay of learning rate to achieve over 90% accuracy on cifar 10.


## Mistake I've made

- Validation and train Set contains duplicate image, which cause the validation accuracy goes insanely high. Showing the inaccurate metrics.
- Not reading the documentation or overly rely on the prompt that AI gives me. Sometimes if you just scraping the curface, it's easy to make stupid mistakes that would cost you like 2 days.
- Using WSL as my first development, which turns out running on linux natively would be signigicantly faster. It seems like even on WSL, it still use CUDA that embedds in windows. And it makes anything related to data transform significantly slower.
- Procrastination. I could've try more thing, or doing more ablation study.

## Things I can improve or try

- Using Kaggle and utilizing the distributed training for getting faster result. (But modularizing the code seems a bit of a pain)
- ~~Using the lab GPU without any permission, for my own greed.~~
- Doing more ablation study. I tend to doing things on the fly and forget what exactly I was aiming for.
- Using tensorboard for realtime visualization. (It feels bad after I'd done the visualizing part and then seeing this.)

## Final Thoughts

For me, implementation plays huge parts for my learning habits. If I cannot implement what I've learned, then I mostly may have not have that level of understanding of the domain knowledge.
