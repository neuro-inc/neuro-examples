Training and Deploying a MNIST Model
====================================

This tutorial is based on [Basic PyTorch example](https://github.com/pytorch/examples/tree/master/mnist).
Our goals here:
1. Prepare and train a basic model on Neu.ro;
2. Wrap the model into an inference HTTP server;
3. Test inference on Neu.ro;
4. Launch production inference on existing Seldon Core.


## Prerequisites
First and foremost, you need to have `neuro` and `neuro-extras` installed in your system:
```shell
# Installing `neuro`
pip install neuromation

# Installing `neuro-extras`
pip install git+https://git@github.com/neuromation/neuro-extras
```

You should also make sure you are logged into Neu.ro:
```
neuro login
```

For deployment on Seldon Core you will also need to have it up and running. It is assumed that you have `kubectl` configured locally to be able to create necessary K8S resources.

## Training

First, we need to copy two files from the mentioned repo:
```shell
curl -O "https://raw.githubusercontent.com/pytorch/examples/master/mnist/main.py"
curl -O "https://raw.githubusercontent.com/pytorch/examples/master/mnist/requirements.txt"
```

If you check the contents of `main.py`, the path to a resulting serialized model is
baked right into the code:
```python
    if args.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
```
This does not look flexible enough for our purposes, thus we would probably need to patch the code and expose another command option to specify the path. For simplicity's sake we won't go there, but rather copy the resulting model once the training process finishes.

The serialized model should be copied to a mounted Storage volume under a chosen path. To accomplish that, we need to write a Dockerfile for the training job and save it as `train.Dockerfile` for further use.
The image will be based on a prebuilt PyTorch image. The complete list can be found on [PyTorch DockerHub](https://hub.docker.com/r/pytorch/pytorch/tags).

```Dockerfile
FROM pytorch/pytorch:1.5-cuda10.1-cudnn7-runtime

ENV MODEL_PATH=/var/storage/model.pkl

COPY main.py .

CMD bash -c "python main.py --save-model; mv mnist_cnn.pt $MODEL_PATH"
```
As you can see, the CMD is meant to perform two operations: (1) train a model, save it in the default location; (2) copy the saved model into another location where we mount our Storage volume.

Lets build the image! The following command does not require a running Docker Engine locally. The build process will be performed on Neu.ro.

```shell
neuro-extras image build -f train.Dockerfile . image:examples/mnist:train
...
INFO: Successfully built image:examples/mnist:train
```
The command above instructs Neu.ro to copy the build context (the current working directory `.` in this case), use the build steps from `train.Dockerfile`, and save the resuling image under `image:examples/mnist:train` in a Neu.ro registry. We can check the registry contents using:

```shell
neuro image tags image:examples/mnist 
image:examples/mnist:train
```

Now that we have the image available within Neu.ro, we can actually run a training job.
```shell
neuro run -s gpu-small -v storage:examples/mnist:/var/storage \
    -e MODEL_PATH=/var/storage/model.pkl image:examples/mnist:train
...
Test set: Average loss: 0.0265, Accuracy: 9910/10000 (99%)
```
Note that we explicitly specify the MODEL_PATH environment variables which leads to the mount Storage volume.
The job takes up to 4 minutes using the `gpu-small` preset.

We can check that there is indeed a serialized model stored on the storage:
```shell
neuro ls -l storage:examples/mnist
-m 4800957 2020-05-04 15:21:18 model.pkl
```

## Deployment

Now that we successfully trained our model, we need to start actually using it.

### Building the Server

`neuro-extras` provides a scaffolding for wrapping the model's code into a functional inference HTTP-server that can be run in Neu.ro, as well as using any other container runtime.

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
The command above creates two files:
1. `seldon_model.py` is a Python module responsible for implemeting the interface that Seldon Core expects. Typically this module reads a serialized model from a mounted Storage volume, deserializes the model and uses it to predict on incoming data points.
2. `seldon.Dockerfile` is a predefined Dockerfile that assembles your code and the Seldon Core inference HTTP-server for further use.

Let's implement the interface first. We should add some missing imports at the top of `seldon_model.py`:
```python
import io

import torch
from PIL import Image
from torchvision import transforms

from .main import Net
```

Then goes the model loading procedure. It should replace the existing dymmy constructor method.
```python
    def __init__(self):
        self._model = Net()
        self._model.load_state_dict(
            torch.load("/storage/model.pkl", map_location=torch.device("cpu"))
        )
        self._model.eval()
```
Note that the code above assumes that the model will reside at the particular path. We will need to mount a Storage volume under `/storage` to make it work. Another interesting detail here is that we'll be running inference on CPU.


Similarly, we need to redefine the `predict` method. As we are dealing with image classification problem in this example, we would like our inference HTTP-server to be able to recieve an image as binary data in `bytes`, predict the classes and return a JSON document with the resulting scores.
```python
    def predict(self, X, features_names):
        data = transforms.ToTensor()(Image.open(io.BytesIO(X)))
        return self._model(data[None, ...]).detach().numpy()
```

We are ready to build the inference image in the same way we built the training one:
```shell
neuro-extras image build -f seldon.Dockerfile . image:examples/mnist:seldon
```

```shell
neuro image tags image:examples/mnist 
image:examples/mnist:train
image:examples/mnist:seldon
```
We see that our registry now has one more image.

### Running the Server on Neu.ro

Before pushing the newly trained and built model to production, let's take a glimpse at how it works within Neu.ro. We need to submit a job that exposes the port `5000` (the default port for the Seldon Core HTTP-server) and mounts a Storage volume with the serialized model to the path mentioned above.
```shell
neuro run -n example-mnist --http 5000 --no-http-auth --detach -v storage:examples/mnist:/storage:ro image:examples/mnist:seldon
```
Note the output of the command above. We should copy the value of the Http URL field to form a `curl` command later.

Once the job is up and running, and the model is loaded, we can start sending some testing data. We'll use a jpg from the MNIST dataset: ![img_103.jpg](./img_103.jpg)

Assuming the Http URL value was `https://example-mnist--user.jobs.neuro-ai-public.org.neu.ro/`, we can now combine a `curl` command. Seldon Core HTTP-server expects binary data sent as the `binData` form field:
```shell
curl -F binData=@img_103.jpg https://example-mnist--user.jobs.neuro-ai-public.org.neu.ro/predict
{"data":{"names":["t:0","t:1","t:2","t:3","t:4","t:5","t:6","t:7","t:8","t:9"],"tensor":{"shape":[1,10],"values":[-3.3279592990875244,-2.659998655319214,-0.39225172996520996,-2.8468592166900635,-5.054018020629883,-5.108893394470215,-4.198001861572266,-2.7248833179473877,-2.8381588459014893,-4.701035499572754]}},"meta":{}}
```
The array element with the index `2` has the highest score, as expected.

If the testing job is no longer needed, we can simply kill it to release the resources:
```shell
neuro kill example-mnist
```

### Running the Server on Seldon Core

After a successfull test of our inference server, we need to push the result into production.

Since your Seldon Core deployment typically resides outside of Neu.ro, for example, in your on-premise K8S cluster, we need to instruct K8S and Seldon to pull our newly-built model image, as well as the corresponding serialized model into the cluster. `neuro-extras` allows creating the required resources easily:

```shell
# Creating a dedicated namespace to keep all the related resources together
kubectl create namespace seldon

# Creating a registry secret for acccessing Neu.ro registry from within K8S
neuro-extras k8s generate-registry-secret | kubectl -n seldon apply -f -

# Creating a secret for accessing Neu.ro (Storage etc) from within K8S
neuro-extras k8s generate-secret | kubectl -n seldon apply -f -

# Creating a Seldon deployment using the URIs we prepared beforehand
neuro-extras seldon generate-deployment image:examples/mnist:seldon \
    storage:examples/mnist/model.pkl | kubectl -n seldon apply -f -
```

At last, let's test our production setup. We will use the same `curl` command, but change the URL. It should lead to your K8S ingress gateway.

```shell
curl -F binData=@img_103.jpg "http://<GATEWAY>/seldon/seldon/neuro-model/api/v1.0/predictions"
{"data":{"names":["t:0","t:1","t:2","t:3","t:4","t:5","t:6","t:7","t:8","t:9"],"tensor":{"shape":[1,10],"values":[-3.3279592990875244,-2.659998655319214,-0.39225172996520996,-2.8468592166900635,-5.054018020629883,-5.108893394470215,-4.198001861572266,-2.7248833179473877,-2.8381588459014893,-4.701035499572754]}},"meta":{}}
```
`2` it is.
