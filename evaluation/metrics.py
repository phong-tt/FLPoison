import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score


def evaluate_multiclass_metrics(model, data_loader, device, num_classes):
    """
    Evaluate a model on a dataloader and return clean multiclass metrics.
    """
    model.eval()
    y_true, y_pred, y_prob = [], [], []
    total_loss, total_samples = 0.0, 0

    with torch.no_grad():
        for images, targets in data_loader:
            images = images.to(device)
            targets = targets.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)

            batch_loss = F.cross_entropy(logits, targets, reduction="sum")
            total_loss += batch_loss.item()
            total_samples += targets.numel()

            y_true.append(targets.detach().cpu().numpy())
            y_pred.append(torch.argmax(logits, dim=1).detach().cpu().numpy())
            y_prob.append(probs.detach().cpu().numpy())

    if total_samples == 0:
        nan = float("nan")
        return {
            "acc": nan,
            "precision": nan,
            "recall": nan,
            "f1": nan,
            "auc": nan,
            "loss": nan,
        }

    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred)
    y_prob = np.concatenate(y_prob)

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    acc = accuracy_score(y_true, y_pred)
    loss = total_loss / total_samples

    try:
        # Ensure probability matrix includes all expected classes.
        if y_prob.shape[1] != num_classes:
            raise ValueError("Probability dimension does not match num_classes.")
        auc = roc_auc_score(y_true, y_prob, multi_class="ovr", average="macro")
    except Exception:
        auc = float("nan")

    return {
        "acc": float(acc),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "auc": float(auc),
        "loss": float(loss),
    }


def aggregate_client_metrics(metric_dicts):
    """
    Aggregate list of per-client metric dicts into mean/std.
    """
    if not metric_dicts:
        return {}

    keys = metric_dicts[0].keys()
    summary = {}
    for key in keys:
        vals = np.array([m.get(key, float("nan")) for m in metric_dicts], dtype=float)
        valid = vals[~np.isnan(vals)]
        if len(valid) == 0:
            mean_val = float("nan")
            std_val = float("nan")
        else:
            mean_val = float(np.mean(valid))
            std_val = float(np.std(valid))
        summary[f"{key}_mean"] = mean_val
        summary[f"{key}_std"] = std_val
    return summary
