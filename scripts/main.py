import os
import subprocess

import mlflow


def run_script(script_cmd, run_name):
    with mlflow.start_run(run_name=run_name, nested=True):
        result = subprocess.run(script_cmd, shell=True, capture_output=True, text=True)
        print(result.stdout)
        if result.stderr:
            print("Errors:", result.stderr)


def main():
    mlflow.set_experiment("Housing_Price_Prediction")

    # Get the root directory of the project
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    print("Started Pipeline at "+str(root_dir))
    if root_dir=="/":
        root_dir=""
    with mlflow.start_run(run_name="Main_Execution"):

        # Step 1: Ingest data
        run_script(
            f"python {os.path.join(root_dir, 'scripts', 'ingest_data.py')} --output-path data/processed --log-level DEBUG",
            "Ingest_Data",
        )

        # Step 2: Train model
        run_script(
            f"python {os.path.join(root_dir, 'scripts', 'train.py')} --input-path data/processed/housing_train.csv "
            "--output-path models --log-level DEBUG",
            "Train_Model",
        )

        # Step 3: Score model
        run_script(
            f"python {os.path.join(root_dir, 'scripts', 'score.py')} --model-path models/housing_model.pkl "
            "--data-path data/processed/housing_test.csv --output-path results --log-level DEBUG",
            "Score_Model",
        )


if __name__ == "__main__":
    main()
