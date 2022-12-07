import argparse
import inspect
import os
import sys

import mlflow
import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import TQDMProgressBar
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.strategies import StrategyRegistry

from utils.codetimer import CodeTimer
from utils.dynamic_class_loading import (
    ensure_data_module_class_exists,
    ensure_model_class_exists,
    get_data_module_class,
    get_model_class,
)

# argument parsing
parser = argparse.ArgumentParser()
# -- model
parser.add_argument(
    "--model",
    type=str,
    required=True,
    help="Class name of the model to train. Make sure there is a corresponding script and class in folder 'models'.",
)
# -- data module
parser.add_argument(
    "--data-module",
    type=str,
    required=True,
    help="Class name of the data module. Make sure there is a corresponding script and class in folder 'data'.",
)
# -- training_strategy
# see https://pytorch-lightning.readthedocs.io/en/stable/extensions/strategy.html#id1 for available strategies and/or
# run:
# from pytorch_lightning.strategies import StrategyRegistry
# StrategyRegistry.available_strategies()
parser.add_argument(
    "--training-strategy",
    type=str,
    required=True,
    choices=StrategyRegistry.available_strategies(),
    help="Training strategy for Lightning.",
)
# -- max-epochs
parser.add_argument("--max-epochs", type=int, default=10, help="Maximum number of epochs to train.")
# -- progress-bar-refresh-rate
parser.add_argument("--progress-bar-refresh-rate", default=100, type=int, help="Refresh rate for training progress bar")
args = parser.parse_args()


# collect some infos about the environment we are running in
# number of nodes
number_nodes = int(os.environ.get("AZUREML_NODE_COUNT", 1))
# number of GPUs/"devices" per node
number_devices_per_node = int(os.environ.get("AZ_BATCHAI_GPU_COUNT", torch.cuda.device_count()))
# are we running in a Compute Cluster?
is_compute_cluster = "AZUREML_RUN_ID" in os.environ
# are we running on the head node?
is_head_node = int(os.environ.get("RANK", 0)) == 0
# IP address
ip_address = os.environ.get("AZ_BATCHAI_NODE_IP", "(not available)")


# say hello and print some context
header_text = f"Let's train model '{args.model}' 🙂"
print(100 * "*")
print(header_text)
print(100 * "*")
print("SCRIPT ARGUMENTS")
print("----------------")
print(f"Model                      : {args.model}")
print(f"Training Strategy          : {args.training_strategy}")
print(f"Maximum Epochs             : {args.max_epochs}")
print(f"Progress Bar Refresh Rate  : {args.progress_bar_refresh_rate}")
print("")
print("ENVIRONMENT")
print("-----------")
print(f"Number Of Nodes            : {number_nodes}")
print(f"Number Of Devices per Node : {number_devices_per_node}")
print(f"Is AzureML Compute Cluster : {is_compute_cluster}")
print(f"Is First Node              : {is_head_node}")
print(f"IP Address                 : {ip_address}")
print("")
print("ENVIRONMENT VARIABLES")
print("---------------------")
for env_var in sorted(os.environ):
    print(f"{env_var}: {os.environ[env_var]}")
print("")
print(100 * "-")
sys.stdout.flush()

# use a local tracking URI for mlflow if we are not running in a Compute Cluster
if not is_compute_cluster:
    mlflow.set_tracking_uri("./@logs/mlruns")

# ensure that the class declarations for the data module and for the model actually exist
ensure_data_module_class_exists(args.data_module)
ensure_model_class_exists(args.model)

# setup trainer
with CodeTimer("Set up trainer"):
    trainer = pl.Trainer(
        accelerator="gpu",
        num_nodes=number_nodes,
        devices=number_devices_per_node,
        max_epochs=args.max_epochs,
        strategy=args.training_strategy,
        logger=pl_loggers.TensorBoardLogger(save_dir=("outputs" if is_compute_cluster else "@logs")),
        callbacks=[
            TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate),
            EarlyStopping(monitor="val_loss", mode="min", patience=10),
        ],
    )

# setup data module
with CodeTimer("Set up data module"):
    data_module_class = get_data_module_class(args.data_module)
    data_module = (
        data_module_class(args)
        if "args" in inspect.signature(data_module_class.__init__).parameters
        else data_module_class()
    )

# setup model
with CodeTimer("Set up model"):
    model_class = get_model_class(args.model)
    model = model_class(args) if "args" in inspect.signature(model_class.__init__).parameters else model_class()

# train model
with mlflow.start_run() as run:
    if is_head_node:
        mlflow.autolog()
        sys.stdout.flush()
    with CodeTimer("Train model") as code_timer:
        trainer.fit(model, data_module)
        net_training_time_wall = code_timer.exit_with_infos()["time_delta_str"]
    if is_head_node:
        mlflow.log_param("net_training_time_wall", net_training_time_wall)

# test model
with CodeTimer("Test model"):
    trainer.test(model, data_module)

# done
print("✔️  Done.")
print("")
