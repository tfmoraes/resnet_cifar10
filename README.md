# Resnet CIFAR10

## Training

```
uv run train.py -m <model>  -b <batch_size> --early-stop <es> --lr <lr>
```

Use --help for its parameters:

```
❯ uv run train.py --help
usage: train.py [-h] [-m MODEL] [-e N] [-b N] [--lr LR] [--early-stop N]

options:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Models: {resnet18, resnet20, resnet34, resnet50, resnet56, resnet101, resnet110} (default: resnet18)
  -e N, --epochs N      number of total epochs to run (default: 200)
  -b N, --batch-size N  Batch size (default: 128)
  --lr LR, --learning-rate LR
                        Learning rate (default: 0.001)
  --early-stop N        Number of epochs of no improvement to early stop. If 0 then early-stop is not activated. (default: 5)

```


## Results

| **Model** | **Optimizer** |   **Scheduler**   | **Learning Rate** | **Batch Size** | **Epochs** |    **Accuracy**    |       **Loss**      |
|:---------:|:-------------:|:-----------------:|:-----------------:|:--------------:|:----------:|:------------------:|:-------------------:|
| ResNet18  | SGD           | CosineAnnealingLR | 0.1               | 256            | 200        | 0.955299973487854  | 0.1743779331445694  |
| ResNet20  | SGD           | CosineAnnealingLR | 0.1               | 256            | 200        | 0.9250999689102173 | 0.28694799542427063 |
| ResNet34  | SGD           | CosineAnnealingLR | 0.1               | 256            | 200        | 0.9539999961853027 | 0.21063581109046936 |
| ResNet50  | SGD           | CosineAnnealingLR | 0.1               | 256            | 200        | 0.9501000046730042 | 0.22970011830329895 |
| ResNet56  | SGD           | CosineAnnealingLR | 0.1               | 256            | 200        | 0.9409999847412109 | 0.29426807165145874 |
| ResNet101 | SGD           | CosineAnnealingLR | 0.1               | 256            | 200        | 0.9535000324249268 | 0.22741803526878357 |
| ResNet110 | SGD           | CosineAnnealingLR | 0.1               | 256            | 200        | 0.9455999732017517 | 0.24942642450332642 |
