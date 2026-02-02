import io
import os
import sys
import time
import unittest
from contextlib import redirect_stderr, redirect_stdout

import pandas as pd

from dotenv import load_dotenv
from automl.automl import AutoML
from tests.base_test import BaseTest
from tests.util import Utils

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def run_single_dataset(dataset_dir, ds_name):
    data_path = os.path.join(dataset_dir, ds_name, f"{ds_name}.data")
    solution_path = os.path.join(dataset_dir, ds_name, f"{ds_name}.solution")
    log_dir = os.path.join(dataset_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{ds_name}.log")

    with open(os.devnull, "w") as devnull, open(log_file, "w") as f, \
         redirect_stdout(devnull), redirect_stderr(devnull):

        automl = AutoML()
        start_time = time.time()

        model, task_type, metrics = automl.run(
            data_path,
            solution_path,
            dataset_name=ds_name,
            train_method="h"
        )

        elapsed_time = time.time() - start_time
        print(f"\n[INFO] Dataset {ds_name} finished in {elapsed_time:.2f}s.", file=f)
        print(f"Results = {metrics}", file=f)

    return {
        "dataset": ds_name,
        "model": model.__class__.__name__,
        "task": task_type,
        "metrics": metrics,
        "time_s": round(elapsed_time, 2),
    }


def sanitize_metric(val):
    """
    Replace negative or None values with 0.0 for display purposes.
    """
    if val is None:
        return 0.0
    if isinstance(val, (int, float)) and val < 0:
        return 0.0
    return val


class TestAutoMLRunAllDatasets(BaseTest):
    @classmethod
    def setUpClass(cls):
        load_dotenv()
        cls.datasets_dir = os.getenv("DATASETS_DIR")
        cls.dataset_names = [
            "data_A", "data_B", "data_C", "data_D", "data_E", "data_G"]
        cls.results = []

    def test_run_all_datasets_sequential(self):
        print("\n\n\n\n\n")
        print("############################################################")
        print("#                                                          #")
        print("#         TEST #3 : Running automl on all datasets         #")
        print("#                                                          #")
        print("############################################################")
        print("\n\n")
        print("[INFO] Running sequentially on all datasets...")

        silent_buffer = io.StringIO()
        with redirect_stdout(silent_buffer), redirect_stderr(silent_buffer):
            for ds in self.dataset_names:
                try:
                    result = run_single_dataset(self.datasets_dir, ds)
                    self.results.append(result)
                except Exception as e:
                    self.fail(f"{ds} failed with error: {e}")

        results_path = os.path.join(self.datasets_dir, "automl_results.csv")
        df_results = pd.DataFrame(self.results)
        df_results.to_csv(results_path, index=False)
        print(f"\n[INFO] Results saved to {results_path}\n")
        print(df_results.to_string(index=False))

        print("\n" + "=" * 100)
        print("LOG SUMMARY BY DATASET (in execution order)")
        print("=" * 100 + "\n")

        logs_dir = os.path.join(self.datasets_dir, "logs")
        for result in self.results:
            log_file = os.path.join(logs_dir, f"{result['dataset']}.log")
            print(f"───────────────────────────── {result['dataset']} ─────────────────────────────")
            try:
                with open(log_file, "r") as f:
                    log_content = f.read()
                    print(log_content.strip())
            except FileNotFoundError:
                print(f"[WARN] No log found for {result['dataset']}.")
            print("\n")

        print("=" * 100)
        print("GLOBAL SUMMARY OF KEY METRICS")
        print("=" * 100)

        rows = []
        for res in self.results:
            metrics = res["metrics"]

            if "accuracy" in metrics:
                val_model = metrics.get("accuracy")
                val_random = metrics.get("random_accuracy")

            elif "r2" in metrics:
                val_model = metrics.get("r2")
                val_random = metrics.get("random_r2")

            else:
                val_model = None
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        val_model = v
                        break
                val_random = None
                for k, v in metrics.items():
                    if "random" in k:
                        val_random = v
                        break

            val_model = sanitize_metric(val_model)
            val_random = sanitize_metric(val_random)

            def format_metric(val):
                try:
                    return f"{float(val):.4f}" if val is not None else "N/A"
                except (ValueError, TypeError):
                    return "N/A"

            rows.append({
                "Dataset": res["dataset"],
                "Model": res["model"],
                "Metric (Model)": format_metric(val_model),
                "Metric (Random)": format_metric(val_random),
            })

        df_table = pd.DataFrame(rows)
        print("\n+----------------+---------------------------+--------------------+--------------------+")
        print("| Dataset        | Model                     | Metric (Model)     | Metric (Random)    |")
        print("+----------------+---------------------------+--------------------+--------------------+")
        for _, r in df_table.iterrows():
            print(f"| {r['Dataset']:<14} | {r['Model']:<25} | {r['Metric (Model)']:>18} | {r['Metric (Random)']:>18} |")
        print("+----------------+---------------------------+--------------------+--------------------+")
        print("=" * 100)
        print("All logs displayed and summary table generated.")
        print("=" * 100)


if __name__ == "__main__":
    unittest.main(verbosity=0)
    Utils.remove_all_pycache()
