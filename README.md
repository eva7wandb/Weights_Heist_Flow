# Objective: Generation of Class Activation Maps (CAM) using grad-CAM method, and overlaying of CAMs on top of misclassified CIFAR10 images

The ResNet18 architecture is used to train the CIFAR10 dataset. The primary goal is to implement
the class activation maps (CAM) for understanding the classification achieved in relation to the
flow of gradients during back-propagation.

Mostly, all of the activities including loading the packages, dataset, model, training
and testing, inferences, will be carried out in a modular scheme while the front end will be Colab
notebook or similar environment.

ResNet18 architecture in case of CIFAR10 dataset starts with a regular 3 x 3 convolution layer
since CIFAR10 images are 32 x 32 only. Other applications of ResNets may start with 7 x 7 or 5 x 5 convolution layers. Then we have custom defined four BasicBlocks, each of them is a pair of residual network connections. Except the first BasicBlock that has a stride of 1 for both the pairs of residual network connections, rest of the BasicBlocks have a stride of 2 for the first residual network connection in their pairs.

The last layer of the fourth BasicBlock connects to Global Average Pooling (GAP) layer and the output is feed to linear layer for 



