import torch
import torch.nn.functional as F


def train_transformer_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)  # (B, S, V)
        lengths = attention_mask.long().sum(dim=1)
        p_pos = lengths - 1
        bsz = input_ids.size(0)
        logits_at_p = logits[torch.arange(bsz, device=device), p_pos, :]

        loss = F.cross_entropy(logits_at_p, labels)
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
def eval_transformer_epoch(model, dataloader, device):
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        logits = model(input_ids, attention_mask)
        lengths = attention_mask.long().sum(dim=1)
        p_pos = lengths - 1
        bsz = input_ids.size(0)
        logits_at_p = logits[torch.arange(bsz, device=device), p_pos, :]

        loss = F.cross_entropy(logits_at_p, labels)
        preds = logits_at_p.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_loss += loss.item() * bsz
        total_count += bsz

    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    acc = total_correct / total_count if total_count > 0 else float('nan')
    return avg_loss, acc
