from pathlib import Path
import sys


def add_repo_path():
    """Ensure local benchmarks/ modules importable from scripts run from repo root."""
    repo_dir = Path(__file__).resolve().parent
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def _make_standard_log(epoch, model_name, task_type, train_metrics, valid_metrics):
    # Ensure consistent keys for wandb across all models. Values may be None
    # when not applicable or not provided by the underlying trainer.
    return {
        "epoch": epoch + 1,
        "model": model_name,
        "train_loss": train_metrics.get("loss") if train_metrics else None,
        "valid_loss": valid_metrics.get("loss") if valid_metrics else None,
        "train_accuracy": train_metrics.get("accuracy") if (train_metrics and task_type == "classification") else None,
        "valid_accuracy": valid_metrics.get("accuracy") if (valid_metrics and task_type == "classification") else None,
        "train_mae": train_metrics.get("mae") if (train_metrics and task_type != "classification") else None,
        "valid_mae": valid_metrics.get("mae") if (valid_metrics and task_type != "classification") else None,
    }


def _make_standard_test_log(model_name, task_type, test_metrics):
    # Standardize test keys for wandb across models
    return {
        "model": model_name,
        "test_loss": test_metrics.get("loss") if test_metrics else None,
        "test_accuracy": test_metrics.get("accuracy") if (test_metrics and task_type == "classification") else None,
        "test_mae": test_metrics.get("mae") if (test_metrics and task_type != "classification") else None,
    }
