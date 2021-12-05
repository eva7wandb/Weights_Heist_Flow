## Work Flow Folder -- Model and utilities files, other supporting files/folders

&nbsp;

The folder structure of this directory in treeformat is given below:

```
.
├── main.py
├── models
│   ├── custom_resnet.py
│   ├── resnet_layerNorm.py
│   └── resnet.py
├── README.md
└── utils
    ├── data.py
    ├── setup.py
    ├── testing.py
    ├── training.py
    └── viz.py

2 directories, 10 files
```

&nbsp;

### File / Folder Details

`main.py` contains the custom class definitions for Model that also has related methods such as load_model, train_model, evaluate_model, and find_lr. It also has functions to gather and show misclassified samples, and to show loss curves.

`models` folder has modules for different architectures used in the implementation of individual assignments such as GradCAMs, One Cycle LR, etc.

`utils.py` has separate files/modules for doing intended tasks such as data.py for class definitions for dataset that includes preprocessing via transforms, dataloader definitions via methods.

`setup.py` sets up the device to cuda or cpu depending upon the availability. It also sets the seed to a value so the results are reproducible. 

`viz.py` includes definitions for model summary, visualization functions for Grad-CAM implementations, and plotting tools for loss.

`training.py` and `testing.py` has function definitions for train (train via backpropagation) and test (getting predictions) procedures performed on batches of samples. It includes the tqdm loader for output results as well. 

&nbsp;

Links to related assignments page:

| Assignment # | Link | Description |
|:-------------|:------------|:-----|
| S8 | [Grad-CAM](https://github.com/eva7wandb/Eva7_Weights_Heist/tree/main/S8) | Grad-CAM implementation on ResNet18 trained model,  |
|  |  | visualize misclassified samples via Grad-CAMs |
| S9 | [One-Cycle LR](https://github.com/eva7wandb/Eva7_Weights_Heist/tree/main/S9) | train a custom ResNet model using One Cycle LR policy |
|  |  |  |
 



