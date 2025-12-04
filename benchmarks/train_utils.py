import torch
import torch.nn.functional as F
from sklearn.metrics import f1_score, mean_absolute_error, mean_squared_error
import numpy as np


def _make_loss_fn_for_transformer(loss_name: str | None):
    if not loss_name:
        return None
    ln = loss_name.lower()
    if ln in ("cross_entropy", "crossentropy", "ce"):
        return lambda logits, labels: torch.nn.functional.cross_entropy(logits, labels)
    if ln in ("mse", "mse_loss"):
        return lambda logits, labels: torch.mean((logits - torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()).pow(2))
    if ln in ("mae", "l1"):
        return lambda logits, labels: torch.mean(torch.abs(logits - torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()))
    if ln in ("rmse",):
        return lambda logits, labels: torch.sqrt(torch.mean((logits - torch.nn.functional.one_hot(labels, num_classes=logits.size(-1)).float()).pow(2)) + 1e-8)
    # default to cross-entropy for transformer classification
    return lambda logits, labels: torch.nn.functional.cross_entropy(logits, labels)


def _make_loss_fn_for_regression(loss_name: str | None):
    """Create loss function for regression tasks."""
    if not loss_name:
        return torch.nn.MSELoss()
    ln = loss_name.lower()
    if ln in ("mae", "l1"):
        return torch.nn.L1Loss()
    if ln in ("mse", "mse_loss"):
        return torch.nn.MSELoss()
    if ln in ("rmse",):
        # RMSE = sqrt(MSE), but we use MSE for loss and sqrt it for metrics
        return torch.nn.MSELoss()
    # default to L1 (MAE) for regression
    return torch.nn.L1Loss()


def train_transformer_epoch(model, dataloader, optimizer, device, loss_name: str | None = None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    all_preds = []
    all_labels = []
    loss_fn = _make_loss_fn_for_transformer(loss_name)
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)  # (B, S, V)
        lengths = attention_mask.long().sum(dim=1)
        # pick the token position for the predicted answer.
        # many of our examples end with `<answer> <eos>`, so the answer
        # token is the second-to-last non-padded token. Clamp to 0
        # to avoid negative indices for very short sequences.
        p_pos = (lengths - 2).clamp(min=0)
        bsz = input_ids.size(0)
        logits_at_p = logits[torch.arange(bsz, device=device), p_pos, :]

        loss = loss_fn(logits_at_p, labels) if loss_fn is not None else F.cross_entropy(logits_at_p, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        preds = logits_at_p.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * bsz
        total_count += bsz
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    acc = total_correct / total_count if total_count > 0 else float('nan')
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if len(all_labels) > 0 else 0.0
    return avg_loss, acc, f1


@torch.no_grad()
def eval_transformer_epoch(model, dataloader, device, loss_name: str | None = None, task_name: str = ""):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_mae = 0.0
    total_count = 0
    all_preds = []
    all_labels = []
    loss_fn = _make_loss_fn_for_transformer(loss_name)
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        lengths = attention_mask.long().sum(dim=1)
        # see comment in training loop: use second-to-last token as answer
        p_pos = (lengths - 2).clamp(min=0)
        bsz = input_ids.size(0)
        logits_at_p = logits[torch.arange(bsz, device=device), p_pos, :]

        loss = loss_fn(logits_at_p, labels) if loss_fn is not None else F.cross_entropy(logits_at_p, labels)
        preds = logits_at_p.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * bsz
        total_count += bsz
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
        # Compute MAE for distance-based tasks like shortest_path
        if "shortest_path" in task_name.lower():
            pred_distances = preds.float()
            label_distances = labels.float()
            total_mae += torch.mean(torch.abs(pred_distances - label_distances)).item() * bsz

    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    acc = total_correct / total_count if total_count > 0 else float('nan')
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0) if len(all_labels) > 0 else 0.0
    mae = total_mae / total_count if total_count > 0 else 0.0
    # Return MAE for shortest_path tasks
    if "shortest_path" in task_name.lower():
        return avg_loss, acc, f1, mae
    return avg_loss, acc, f1


def train_regression_epoch(model, dataloader, optimizer, device, loss_name: str | None = None):
    """Training loop for regression transformers (e.g., ZINC)."""
    model.train()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    all_preds = []
    all_labels = []
    
    loss_fn = _make_loss_fn_for_regression(loss_name)
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)  # (B, S, output_dim)
        
        # Extract prediction at appropriate position
        lengths = attention_mask.long().sum(dim=1)
        p_pos = (lengths - 2).clamp(min=0)
        bsz = input_ids.size(0)
        
        # Handle both sequence and non-sequence outputs
        if logits.dim() == 3:  # (B, S, output_dim)
            preds = logits[torch.arange(bsz, device=device), p_pos, :]
        else:  # (B, output_dim)
            preds = logits
        
        # Ensure labels are correct shape
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        if preds.dim() > labels.dim():
            preds = preds.squeeze(-1) if preds.shape[-1] == 1 else preds
        
        # Compute loss and metrics
        loss = loss_fn(preds, labels)
        mae = torch.mean(torch.abs(preds - labels)).item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * bsz
        total_mae += mae * bsz
        total_count += bsz
        all_preds.extend(preds.detach().cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    avg_mae = total_mae / total_count if total_count > 0 else float('nan')
    
    # Compute RMSE
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2)) if len(all_labels) > 0 else float('nan')
    
    return avg_loss, avg_mae, rmse


@torch.no_grad()
def eval_regression_epoch(model, dataloader, device, loss_name: str | None = None):
    """Evaluation loop for regression transformers (e.g., ZINC)."""
    model.eval()
    total_loss = 0.0
    total_mae = 0.0
    total_count = 0
    all_preds = []
    all_labels = []
    
    loss_fn = _make_loss_fn_for_regression(loss_name)
    
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        
        # Forward pass
        logits = model(input_ids, attention_mask)  # (B, S, output_dim)
        
        # Extract prediction at appropriate position
        lengths = attention_mask.long().sum(dim=1)
        p_pos = (lengths - 2).clamp(min=0)
        bsz = input_ids.size(0)
        
        # Handle both sequence and non-sequence outputs
        if logits.dim() == 3:  # (B, S, output_dim)
            preds = logits[torch.arange(bsz, device=device), p_pos, :]
        else:  # (B, output_dim)
            preds = logits
        
        # Ensure labels are correct shape
        if labels.dim() == 0:
            labels = labels.unsqueeze(0)
        if preds.dim() > labels.dim():
            preds = preds.squeeze(-1) if preds.shape[-1] == 1 else preds
        
        # Compute loss and metrics
        loss = loss_fn(preds, labels)
        mae = torch.mean(torch.abs(preds - labels)).item()
        
        total_loss += loss.item() * bsz
        total_mae += mae * bsz
        total_count += bsz
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    avg_mae = total_mae / total_count if total_count > 0 else float('nan')
    
    # Compute RMSE
    all_preds = np.array(all_preds).flatten()
    all_labels = np.array(all_labels).flatten()
    rmse = np.sqrt(np.mean((all_preds - all_labels) ** 2)) if len(all_labels) > 0 else float('nan')
    
    return avg_loss, avg_mae, rmse

