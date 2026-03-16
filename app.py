# The data set used in this example is from http://archive.ics.uci.edu/ml/datasets/Wine+Quality
# P. Cortez, A. Cerdeira, F. Almeida, T. Matos and J. Reis.
# Modeling wine preferences by data mining from physicochemical properties. In Decision Support Systems, Elsevier, 47(4):547-553, 2009.

import os
import warnings
import sys
import tempfile
import pickle
import dagshub
dagshub.init(repo_owner='vignesh-DA', repo_name='mlflow', mlflow=True)

import mlflow

import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import ElasticNet
import mlflow
from mlflow.models import infer_signature
import mlflow.sklearn

import logging

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)
    experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "wine-quality")
    mlflow.set_experiment(experiment_name)

    print(f"MLflow Tracking URI: {mlflow.get_tracking_uri()}")
    print(f"MLflow Experiment: {experiment_name}")

    # Read the wine-quality csv file from the URL
    csv_url = (
        "https://raw.githubusercontent.com/mlflow/mlflow/master/tests/datasets/winequality-red.csv"
    )
    try:
        data = pd.read_csv(csv_url, sep=";")
    except Exception as e:
        logger.exception(
            "Unable to download training & test CSV, check your internet connection. Error: %s", e
        )
        raise

    # Split the data into training and test sets. (0.75, 0.25) split.
    train, test = train_test_split(data, test_size=0.25, random_state=42)

    # The predicted column is "quality" which is a scalar from [3, 9]
    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
    l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

    with mlflow.start_run(run_name="train-elasticnet"):
        lr = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42)
        lr.fit(train_x, train_y)

        predicted_qualities = lr.predict(test_x)

        (rmse, mae, r2) = eval_metrics(test_y, predicted_qualities)

        print("Elasticnet model (alpha={:f}, l1_ratio={:f}):".format(alpha, l1_ratio))
        print("  RMSE: %s" % rmse)
        print("  MAE: %s" % mae)
        print("  R2: %s" % r2)

        mlflow.log_param("alpha", alpha)
        mlflow.log_param("l1_ratio", l1_ratio)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        mlflow.log_metric("mae", mae)

        # Save train, test and model pickle as artifacts in MLflow (DagsHub)
        with tempfile.TemporaryDirectory() as temp_dir:
            train_csv_path = os.path.join(temp_dir, "train_data.csv")
            train.to_csv(train_csv_path, index=False)
            mlflow.log_artifact(train_csv_path, artifact_path="data")

            test_csv_path = os.path.join(temp_dir, "test_data.csv")
            test.to_csv(test_csv_path, index=False)
            mlflow.log_artifact(test_csv_path, artifact_path="data")

            model_pkl_path = os.path.join(temp_dir, "model.pkl")
            with open(model_pkl_path, "wb") as model_file:
                pickle.dump(lr, model_file)
            mlflow.log_artifact(model_pkl_path, artifact_path="model")

        # Also log the model using MLflow sklearn format
        signature = infer_signature(train_x, lr.predict(train_x))
        mlflow.sklearn.log_model(lr, "elasticnet_model", signature=signature)

    # Find and save/log the best model from all runs in the experiment
    print("\n" + "="*50)
    print("Finding and saving the BEST MODEL...")
    print("="*50)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        order_by=["metrics.r2 DESC"],
        max_results=100,
    )

    if len(runs) > 0 and "metrics.r2" in runs.columns:
        runs = runs.dropna(subset=["metrics.r2"])

    if len(runs) > 0:
        best_run = runs.iloc[0]
        best_run_id = best_run["run_id"]
        best_r2 = best_run["metrics.r2"]

        print(f"\nBest Run ID: {best_run_id}")
        print(f"Best R2 Score: {best_r2}")
        print(f"Best RMSE: {best_run['metrics.rmse']}")
        print(f"Best MAE: {best_run['metrics.mae']}")
        print(f"Alpha: {best_run['params.alpha']}")
        print(f"L1 Ratio: {best_run['params.l1_ratio']}")

        # Load best model and log it in a dedicated MLflow run (saved to DagsHub)
        best_model_uri = f"runs:/{best_run_id}/elasticnet_model"
        best_model = mlflow.sklearn.load_model(best_model_uri)

        with mlflow.start_run(run_name="best-model-sync"):
            mlflow.set_tag("source_best_run_id", best_run_id)
            mlflow.log_param("best_alpha", best_run["params.alpha"])
            mlflow.log_param("best_l1_ratio", best_run["params.l1_ratio"])
            mlflow.log_metric("best_r2", float(best_r2))
            mlflow.log_metric("best_rmse", float(best_run["metrics.rmse"]))
            mlflow.log_metric("best_mae", float(best_run["metrics.mae"]))

            with tempfile.TemporaryDirectory() as temp_dir:
                best_model_path = os.path.join(temp_dir, "best_model.pkl")
                with open(best_model_path, "wb") as best_model_file:
                    pickle.dump(best_model, best_model_file)
                mlflow.log_artifact(best_model_path, artifact_path="best/model")

                train_csv_path = os.path.join(temp_dir, "train_data.csv")
                test_csv_path = os.path.join(temp_dir, "test_data.csv")
                train.to_csv(train_csv_path, index=False)
                test.to_csv(test_csv_path, index=False)
                mlflow.log_artifact(train_csv_path, artifact_path="best/data")
                mlflow.log_artifact(test_csv_path, artifact_path="best/data")

            best_signature = infer_signature(train_x, best_model.predict(train_x))
            mlflow.sklearn.log_model(best_model, "best_model", signature=best_signature)

        print("\n✓ Best model and split data were logged to DagsHub MLflow experiment")
        print("="*50)
    else:
        print("No completed runs with R2 metric found in this experiment yet.")


