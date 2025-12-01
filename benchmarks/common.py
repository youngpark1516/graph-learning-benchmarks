from pathlib import Path
import sys


def add_repo_path():
    """Ensure local benchmarks/ modules importable from scripts run from repo root."""
    repo_dir = Path(__file__).resolve().parent
    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))


def _make_standard_log(epoch, model_name, task_type, train_metrics, valid_metrics, eval_metrics=None):
    """Build a standardized training/validation log dict for wandb.

    If `eval_metrics` is provided (list of metric names), those metrics will be
    included for both train and valid logs (e.g. ['mae','accuracy']). If not
    provided, previous behavior applies: report `accuracy` for
    `classification` tasks and `mae` for regression tasks.
    
    Always includes f1_score for classification tasks.
    """
    out = {
        "epoch": epoch + 1,
        "model": model_name,
        "train_loss": train_metrics.get("loss") if train_metrics else None,
        "valid_loss": valid_metrics.get("loss") if valid_metrics else None,
    }

    # If eval_metrics provided, include each requested metric.
    if eval_metrics:
        for m in eval_metrics:
            m_lower = m.lower()
            out[f"train_{m_lower}"] = train_metrics.get(m_lower) if train_metrics else None
            out[f"valid_{m_lower}"] = valid_metrics.get(m_lower) if valid_metrics else None
        # Also always include f1_score for classification tasks
        if task_type == "classification":
            out[f"train_f1_score"] = train_metrics.get("f1_score") if train_metrics else None
            out[f"valid_f1_score"] = valid_metrics.get("f1_score") if valid_metrics else None
        return out

    # Backwards compatible defaults when no explicit eval_metrics given
    out["train_accuracy"] = train_metrics.get("accuracy") if (train_metrics and task_type == "classification") else None
    out["valid_accuracy"] = valid_metrics.get("accuracy") if (valid_metrics and task_type == "classification") else None
    out["train_f1_score"] = train_metrics.get("f1_score") if (train_metrics and task_type == "classification") else None
    out["valid_f1_score"] = valid_metrics.get("f1_score") if (valid_metrics and task_type == "classification") else None
    out["train_mae"] = train_metrics.get("mae") if (train_metrics and task_type != "classification") else None
    out["valid_mae"] = valid_metrics.get("mae") if (valid_metrics and task_type != "classification") else None
    return out


def _make_standard_test_log(model_name, task_type, test_metrics, eval_metrics=None):
    """Build a standardized test log dict for wandb.

    If `eval_metrics` is provided, include those metrics as `test_<metric>`.
    Otherwise preserve the previous behavior based on `task_type`.
    
    Always includes f1_score for classification tasks.
    """
    out = {
        "model": model_name,
        "test_loss": test_metrics.get("loss") if test_metrics else None,
    }

    if eval_metrics:
        for m in eval_metrics:
            out[f"test_{m.lower()}"] = test_metrics.get(m.lower()) if test_metrics else None
        # Also always include f1_score for classification tasks
        if task_type == "classification":
            out[f"test_f1_score"] = test_metrics.get("f1_score") if test_metrics else None
        return out

    out["test_accuracy"] = test_metrics.get("accuracy") if (test_metrics and task_type == "classification") else None
    out["test_f1_score"] = test_metrics.get("f1_score") if (test_metrics and task_type == "classification") else None
    out["test_mae"] = test_metrics.get("mae") if (test_metrics and task_type != "classification") else None
    return out
