"""Command-line interface for training the CNN model."""
import logging
import os
import sys
from datetime import datetime

import click
import numpy as np
import torch

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

# Add project root to sys.path to allow for src imports
sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
)
# pylint: disable=wrong-import-position
from src.model.model import (  # noqa: E402
    CNNModel,
    train_model,
    save_model,
)


@click.command()
@click.option("--train-images-path", required=True, type=str)
@click.option("--train-labels-path", required=True, type=str)
@click.option("--output-model-path", required=True, type=str)
@click.option("--epochs", default=5, type=int)
@click.option("--learning-rate", default=0.001, type=float)
@click.option("--batch-size", default=32, type=int)
@click.option("--val-split", default=0.2, type=float)
@click.option(
    "--num-classes", default=10, type=int, help="Number of output classes"
)
@click.option(
    "--input-shape",
    default="1,28,28",
    type=str,
    help="Input shape comma-separated",
)
@click.option(
    "--wandb-project", default=None, type=str, help="W&B project name"
)
@click.option("--wandb-entity", default=None, type=str, help="W&B entity name")
@click.option(
    "--wandb-tags", default="", type=str, help="W&B tags, comma-separated"
)
@click.option(
    "--wandb-name-prefix", default="run", type=str, help="W&B run name prefix"
)
@click.option(
    "--wandb-enabled",
    default='false',
    type=str,
    help="Enable W&B logging ('true' or 'false')",
)
def main(  # pylint: disable=too-many-arguments
    train_images_path,
    train_labels_path,
    output_model_path,
    epochs,
    learning_rate,
    batch_size,
    val_split,
    num_classes,
    input_shape,
    wandb_project,
    wandb_entity,
    wandb_tags,
    wandb_name_prefix,
    wandb_enabled,
):
    """
    Train a CNN model on the given dataset.

    This function takes paths to training data, model configuration parameters,
    and optional wandb settings, then trains and saves a CNN model.
    """
    logging.basicConfig(level=logging.INFO)
    logging.info("✅ Model training started")

    # Convert wandb_enabled from string to boolean
    is_wandb_enabled = wandb_enabled.lower() in ('true', '1', 'yes')

    # Initialize W&B if enabled
    if is_wandb_enabled:
        if WANDB_AVAILABLE:
            # Generate a unique run name with a timestamp
            timestamp = datetime.now().strftime('%Y%m%d-%H%M%S')
            run_name = f"{wandb_name_prefix}-{timestamp}"

            # Parse input shape from string
            parsed_input_shape = [int(x) for x in input_shape.split(',')]

            wandb.init(
                project=wandb_project,
                entity=wandb_entity,
                name=run_name,
                tags=wandb_tags.split(',') if wandb_tags else None,
                config={
                    "epochs": epochs,
                    "learning_rate": learning_rate,
                    "batch_size": batch_size,
                    "val_split": val_split,
                    "num_classes": num_classes,
                    "input_shape": parsed_input_shape,
                    "save_path": output_model_path,
                },
            )
            logging.info("W&B initialized for run: %s", run_name)
        else:
            logging.warning(
                "WandB logging was enabled, but wandb is not "
                "installed. Skipping."
            )
            is_wandb_enabled = False

    x = np.load(train_images_path)
    y = np.load(train_labels_path)

    x_tensor = torch.tensor(x, dtype=torch.float32).permute(0, 3, 1, 2)
    y_tensor = torch.tensor(np.argmax(y, axis=1), dtype=torch.long)

    dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    model = CNNModel(num_classes=num_classes)
    config = {
        "model": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "val_split": val_split,
        }
    }
    model = train_model(model, loader, config)

    save_model(model, output_model_path)
    logging.info("✅ Model training complete and saved")

    if is_wandb_enabled and WANDB_AVAILABLE and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
