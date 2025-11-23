import torch
import torch.nn.functional as F


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


def train_transformer_epoch(model, dataloader, optimizer, device, loss_name: str | None = None):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    loss_fn = _make_loss_fn_for_transformer(loss_name)
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)  # (B, S, V)
        lengths = attention_mask.long().sum(dim=1)
        p_pos = lengths - 1
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

    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    acc = total_correct / total_count if total_count > 0 else float('nan')
    return avg_loss, acc


@torch.no_grad()
def eval_transformer_epoch(model, dataloader, device, loss_name: str | None = None):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    loss_fn = _make_loss_fn_for_transformer(loss_name)
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        lengths = attention_mask.long().sum(dim=1)
        p_pos = lengths - 1
        bsz = input_ids.size(0)
        logits_at_p = logits[torch.arange(bsz, device=device), p_pos, :]

        loss = loss_fn(logits_at_p, labels) if loss_fn is not None else F.cross_entropy(logits_at_p, labels)
        preds = logits_at_p.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * bsz
        total_count += bsz

    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    acc = total_correct / total_count if total_count > 0 else float('nan')
    return avg_loss, acc
