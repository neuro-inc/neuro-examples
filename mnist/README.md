Training and Deploying a MNIST Model
====================================

Based on [Basic PyTorch example](https://github.com/pytorch/examples/tree/master/mnist).


## Prerequisites
```shell
# Installing `neuro`
pip install neuromation
neuro login

# Installing `neuro-extras`
pip install git+ssh://git@github.com/neuromation/neuro-extras.git
```

## Training

First, we need to copy two files from the mentioned repo:
```shell
curl -O "https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py"
curl -O "https://raw.githubusercontent.com/pytorch/examples/master/mnist/requirements.txt"
```

As you can see in `main.py`, the path to a resulting serialized tensor is
baked right into the code:
```python
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
```

Upon finishing training, the tensor should be copied to a mounted volume under
a chosen name.

To accomplish that, we need to write a Dockerfile for the training job.
The image will be based on a prebuilt PyTorch image. The complete list can be
found on [PyTorch DockerHub](https://hub.docker.com/r/pytorch/pytorch/tags).

```Dockerfile
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

ENV MODEL_PATH=/var/storage/model.pkl

COPY main.py .

CMD bash -c "python main.py --save-model; mv mnist_cnn.pt $MODEL_PATH"
```

Lets build an image! The following command does not require running Docker
Engine locally. The build process will be performed on the platform.

```shell
neuro-extras image build -f train.Dockerfile . image:examples/mnist:train
...
INFO: Successfully built image:examples/mnist:train
```

Now that we have the image available within the platform, we can actually run
a training job.
```shell
neuro run -s gpu-small -v storage:examples/mnist:/var/storage \
    -e MODEL_PATH=/var/storage/model.pkl image:examples/mnist:train
...
Test set: Average loss: 0.0265, Accuracy: 9910/10000 (99%)
```
The job takes up to 4 minutes using the `gpu-small` preset.

We can check that there is indeed a serialized tensor stored on the storage:
```shell
neuro ls -l storage:examples/mnist                                                                                                                                                                                                                    master ◼
-m 4800957 2020-05-04 15:21:18 model.pkl
```

## Deployment

```shell
neuro-extras seldon init-package .

ls -l
-rw-r--r--  1 user  group  2375 May  4 15:33 README.md
-rw-r--r--  1 user  group  5136 May  4 14:36 main.py
-rw-r--r--  1 user  group    18 May  4 14:37 requirements.txt
-rwx------  1 user  group   517 Apr 30 18:21 seldon.Dockerfile
-rw-r--r--  1 user  group  1193 Apr 30 18:22 seldon_model.py
-rw-r--r--  1 user  group   176 May  4 15:12 train.Dockerfile
```
