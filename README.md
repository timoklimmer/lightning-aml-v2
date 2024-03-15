# Lightning on Azure Machine Learning v2 using Nebula checkpoints

Demonstrates the use of PyTorch Lightning on Azure Machine Learning. SDK/CLI v2, Nebula checkpointing.

![Azure ML Screenshot](./repo/media/aml-screenshot.png)

Usage:

1. Get an Azure ML workspace.

2. Ensure that you have got the Azure CLI and the corresponding ml extension installed on your machine. Alternatively,
   you can use an Azure ML Compute Instance as your dev machine. This gives the advantage that you can develop on a
   machine that has a maybe required GPU model built in. If you don't have an Ubuntu system at hand, it's also a good
   means to get access to an Ubuntu box.

3. Clone the repo and open the code in an editor of your choice. For best usability, VS.Code or VS.Code for Web are
   recommended, though.

4. Customization:
   - For a custom dataset, place a LightningDataModule script into folder 'data' (similar to MNISTfromBlob.py).
   - For a custom model, place a LightningModule script into folder 'models' (similar to CustomModel.py).
   - To support model/data module authoring, you can setup a Python environment with all packages required by running
     `conda env create -n laml --file conda-dev.yaml`.
   - When submitting a job to Azure ML, you may need to adjust the corresponding `train_*.yaml` file before. especially,
     you may also need to create an Azure ML compute cluster and/or create a custom environment.
   - Adjust the `train_*.yaml` files to include your file path to the dataset, name of the compute cluster and cluster
     capacity. You may also want to adjust the parameters such as batch size and max epochs. See included comments.

5. Run any of the `train_*.yaml` files by clicking the Azure ML icon on top right in VS.Code to train a model in
   AzureML. Note that you need the Azure Machine Learning extension installed in VS.Code to make this button available. Alternatively, run a `az ml job create -g <resource_group> -w <workspace> --file <yaml file> --stream` command in
   your command line.
   
   Please note that Nebula currently only exists in curated ACPT environment on Azure therefore it won't be possible to
   run scripts without using one of the ACPT containers. Documentation on ACPT containers including best practices can
   be found [here](https://github.com/Azure/azureml-examples/blob/main/best-practices/largescale-deep-learning/Environment/ACPT.md).

   To run the script with no customized dataset provided, please modify the `train_*.yaml` files to include
   MNISTfromMNIST as the data-module and ImageClassifier as the model. Also please comment out the data_folder and
   batch_size parameters. In this way, the inbuilt MNIST dataset will be used.

7. Open Azure ML Studio / Jobs and watch the training job progress.

The code provided uses mlflow autologging and [Nebula checkpointing](https://github.com/MicrosoftDocs/azure-docs/blob/main/articles/machine-learning/reference-checkpoint-performance-for-large-models.md).

As always, feel free to use but don't blame me/us if things go wrong. 
