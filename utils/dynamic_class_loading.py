import importlib
import os

import pytorch_lightning as pl


def ensure_data_module_class_exists(data_module_name):
    """Throw an exception if there are issues with loading a data module class."""
    # script file exists
    if not os.path.exists(f"data/{data_module_name}/{data_module_name}.py"):
        raise ValueError(
            (
                f"To use data module '{data_module_name}', a Python script named '{data_module_name}.py' needs to be "
                f"contained in folder 'data/{data_module_name}'. Ensure that such file exists."
            )
        )

    # script file has a class with the name of the data module
    data_module_module = importlib.import_module(f"data.{data_module_name}.{data_module_name}")
    if not hasattr(data_module_module, data_module_name):
        raise ValueError(
            (
                f"The file at 'data/{data_module_name}/{data_module_name}.py' does not declare a class named "
                f"'{data_module_name}'. Ensure that this class is declared."
            )
        )

    # data module class is a lightning data module
    data_module_class = getattr(data_module_module, data_module_name)
    if not issubclass(data_module_class, pl.LightningDataModule):
        raise ValueError(
            (
                f"The given class for data module '{data_module_name}' is not inheriting from class "
                f"'LightningDataModule'. Ensure that the class declaring your data module is derived from class "
                f"'LightningDataModule'."
            )
        )


def ensure_model_class_exists(model_name):
    """Throw an exception if there are issues with loading a model's class."""
    # script file exists
    if not os.path.exists(f"models/{model_name}/{model_name}.py"):
        raise ValueError(
            (
                f"To train model '{model_name}', a Python script named '{model_name}.py' needs to be contained in "
                f"folder 'models/{model_name}'. Ensure that such file exists."
            )
        )

    # script file has a class with the name of the model
    model_module = importlib.import_module(f"models.{model_name}.{model_name}")
    if not hasattr(model_module, model_name):
        raise ValueError(
            (
                f"The file at 'models/{model_name}/{model_name}.py' does not declare a class named '{model_name}'. "
                f"Ensure that this class is declared."
            )
        )

    # model class is a lightning module
    model_class = getattr(model_module, model_name)
    if not issubclass(model_class, pl.LightningModule):
        raise ValueError(
            (
                f"The given class for model '{model_name}' is not inheriting from class 'LightningModule'. "
                f"Ensure that the class declaring your model is derived from class 'LightningModule'."
            )
        )


def get_data_module_class(data_module_name):
    """Return the class for the given data module name."""
    return getattr(importlib.import_module(f"data.{data_module_name}.{data_module_name}"), data_module_name)


def get_model_class(model_name):
    """Return the class for the given model name."""
    return getattr(importlib.import_module(f"models.{model_name}.{model_name}"), model_name)
