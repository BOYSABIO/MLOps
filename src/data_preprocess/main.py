import os
import click
import numpy as np
from data_preprocessing import preprocess_data, save_preprocessed_data

@click.command()
@click.option("--input-path", required=True, type=click.Path(exists=True))
@click.option("--output-path", default="data/processed", type=click.Path())
def main(input_path, output_path):
    try:
        x_train = np.load(os.path.join(input_path, "x_train.npy"))
        y_train = np.load(os.path.join(input_path, "y_train.npy"))
        x_test = np.load(os.path.join(input_path, "x_test.npy"))
        y_test = np.load(os.path.join(input_path, "y_test.npy"))
    except Exception as e:
        raise FileNotFoundError("Expected .npy files not found in input path.") from e

    x_train_prep, y_train_prep = preprocess_data(x_train, y_train)
    x_test_prep, y_test_prep = preprocess_data(x_test, y_test)

    save_preprocessed_data(x_train_prep, y_train_prep, output_path, data_type="train")
    save_preprocessed_data(x_test_prep, y_test_prep, output_path, data_type="test")

    print("✅ Data preprocessing completed.")

if __name__ == "__main__":
    main()