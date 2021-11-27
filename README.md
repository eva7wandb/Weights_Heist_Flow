### Objective: Generation of Class Activation Maps (CAM) using grad-CAM method,
### and overlaying of CAMs on top of misclassified CIFAR10 images

The ResNet18 architecture is used to train the CIFAR10 dataset. The primary goal is to implement
the class activation maps (CAM) for understanding the classification achieved in relation to the
flow of gradients during back-propagation.

Mostly, all of the activities including loading the packages, dataset, model, training
and testing, inferences, will be carried out in a modular scheme while the front end will be Colab
notebook or similar environment. The resnet model is saved under model directory. The full directory
structure in tree form is given below:

```
.
├── eva7S8_tree
├── main.py
├── models
│   ├── resnet_layerNorm.py
│   └── resnet.py
├── README.md
├── S8_resnetLayerNorm.ipynb
└── utils
    ├── data.py
    ├── setup.py
    ├── testing.py
    ├── training.py
    └── viz.py
```

ResNet18 architecture in case of CIFAR10 dataset starts with a regular 3 x 3 convolution layer
since CIFAR10 images are 32 x 32 only. Other applications of ResNets may start with 7 x 7 or 5 x 5 
convolution layers. Then we have custom defined four BasicBlocks, each of them is a pair of residual
network connections. Except the first BasicBlock that has a stride of 1 for both the pairs of residual
network connections, rest of the BasicBlocks have a stride of 2 for the first residual network
connection in their pairs.

The last layer of the fourth BasicBlock connects to Global Average Pooling (GAP) layer and the output
(batch_size, 512, 1, 1) is fed to linear layer that yields a vector of 10 classes.

The class activation maps is generated using the the gradient-weighted CAM (grad-CAM) method which
computes gradients of the interested class for the previous layer as in back-propagation, and then
compute the amplified output w.r.t to the corresponding gradient of the respective channels of the
feature maps.

Finally the grad-CAM generated class activation maps also called the heat-map of the layer is superimposed
on the original image supplied, and this is carried batchwise thus yielding class-activated images. This
can be very useful in understanding where the neural network is looking at the input image.

Below is the colab output showing the epochs, and respective loss and accuracy metrics.



