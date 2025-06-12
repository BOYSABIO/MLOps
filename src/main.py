import mlflow
import argparse
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--step",
        default="all",
        help="Step to run: all | data_load | data_validation | data_preprocess | model | evaluation | features | inference"
    )
    args = parser.parse_args()

    raw_path = os.path.abspath("data/raw")
    processed_path = os.path.abspath("data/processed")
    model_path = os.path.abspath("models/model.pth")
    reports_path = os.path.abspath("reports/embeddings")
    predictions_path = os.path.abspath("predictions/prediction.txt")

    train_images = os.path.join(processed_path, "train_images.npy")
    train_labels = os.path.join(processed_path, "train_labels.npy")
    test_images = os.path.join(processed_path, "test_images.npy")
    test_labels = os.path.join(processed_path, "test_labels.npy")

    if args.step in ("all", "data_load"):
        mlflow.run(
            uri="src/data_load",
            entry_point="main",
            parameters={"output_path": raw_path},
        )

    if args.step in ("all", "data_validation"):
        mlflow.run(
            uri="src/data_validation",
            entry_point="main",
            parameters={"input_path": raw_path},
        )

    if args.step in ("all", "data_preprocess"):
        mlflow.run(
            uri="src/data_preprocess",
            entry_point="main",
            parameters={
                "input_path": raw_path,
                "output_path": processed_path
            },
        )

    if args.step in ("all", "model"):
        mlflow.run(
            uri="src/model",
            entry_point="main",
            parameters={
                "train-images-path": train_images,
                "train-labels-path": train_labels,
                "output-model-path": model_path,
                "epochs": 5,
                "learning-rate": 0.001,
                "batch-size": 32
            }
        )

    if args.step in ("all", "evaluation"):
        mlflow.run(
            uri="src/evaluation",
            entry_point="main",
            parameters={
                "model-path": model_path,
                "test-images-path": test_images,
                "test-labels-path": test_labels
            }
        )

    if args.step in ("all", "features"):
        mlflow.run(
            uri="src/features",
            entry_point="main",
            parameters={
                "model-path": model_path,
                "test-images-path": test_images,
                "test-labels-path": test_labels,
                "output-dir": reports_path
            }
        )

    if args.step in ("all", "inference"):
        mlflow.run(
            uri="src/inference",
            entry_point="main",
            parameters={
                "model-path": model_path,
                "image-path": test_images,
                "output-path": predictions_path
            }
        )


if __name__ == "__main__":
    main()
