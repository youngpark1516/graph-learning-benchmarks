#!/usr/bin/env python3
# coding=utf-8

"""Transformer model for text-only up to <p>, predicting the token after <p>."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR


class GraphDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        task: str,
        algorithm: str,
        split: str,
        max_seq_length: int = 512,
        n_samples_per_file: int = -1,
        sampling_seed: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.task = task
        self.algorithm = algorithm
        self.split = split
        self.max_seq_length = max_seq_length
        self.n_samples_per_file = n_samples_per_file
        self.sampling_seed = sampling_seed

        self.samples = self._load_samples()
        self.token2idx = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2, "[UNK]": 3}
        self._build_vocabulary()
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)

    def _load_samples(self) -> List[dict]:
        task_dir = self.data_dir / "tasks_autograph" / self.task / self.algorithm / self.split
        samples = []
        rng = random.Random(self.sampling_seed) if self.sampling_seed is not None else random
        for json_file in sorted(task_dir.glob("*.json")):
            with json_file.open() as f:
                file_samples = json.load(f)
                if isinstance(file_samples, list) and self.n_samples_per_file and self.n_samples_per_file > 0:
                    k = min(len(file_samples), int(self.n_samples_per_file))
                    # sample without replacement
                    chosen = rng.sample(file_samples, k)
                    samples.extend(chosen)
                else:
                    samples.extend(file_samples)
        return samples

    def _build_vocabulary(self):
        for sample in self.samples:
            for t in sample["text"].strip().split():
                if t not in self.token2idx:
                    self.token2idx[t] = len(self.token2idx)

    def _text_until_p(self, text_tokens: List[str]) -> Tuple[List[str], str]:
        try:
            i_p = text_tokens.index("<p>")
        except ValueError:
            raise ValueError("Text must contain <p>.")
        if not (i_p < len(text_tokens) - 1):
            raise ValueError("Need one token after <p>.")
        prefix = text_tokens[: i_p + 1]
        answer_tok = text_tokens[i_p + 1]
        return prefix, answer_tok

    def _to_id(self, tok: str) -> int:
        return self.token2idx.get(tok, self.token2idx["[UNK]"])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        text_tokens = sample["text"].strip().split()

        prefix, answer_tok = self._text_until_p(text_tokens)
        combined = prefix

        if len(combined) > self.max_seq_length - 1:
            combined = combined[: self.max_seq_length - 1]

        token_ids = [self.token2idx["[CLS]"]]
        token_ids.extend([self._to_id(t) for t in combined])

        padding_len = self.max_seq_length - len(token_ids)
        if padding_len > 0:
            token_ids.extend([self.token2idx["[PAD]"]] * padding_len)

        attention_mask = [1 if t != self.token2idx["[PAD]"] else 0 for t in token_ids]
        label_id = self._to_id(answer_tok)

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            "label": torch.tensor(label_id, dtype=torch.long),
            "graph_id": sample.get("graph_id", ""),
        }


class TransformerEncoder(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=False)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x


class GraphTransformer(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 6,
        d_ff: int = 1024,
        dropout: float = 0.1,
        max_seq_length: int = 512,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.randn(max_seq_length, d_model))
        self.encoder_layers = nn.ModuleList(
            [TransformerEncoder(d_model, n_heads, d_ff, dropout) for _ in range(n_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, S = input_ids.size()
        x = self.token_embedding(input_ids)
        x = x + self.position_embedding[:S].unsqueeze(0)
        x = self.dropout(x)
        key_padding_mask = (attention_mask == 0) if attention_mask is not None else None
        x = x.transpose(0, 1)
        for layer in self.encoder_layers:
            x = layer(x, mask=key_padding_mask)
        x = x.transpose(0, 1)
        logits = self.output_projection(x)
        return logits


def _step_at_p(logits: torch.Tensor, attention_mask: torch.Tensor, labels: torch.Tensor):
    lengths = attention_mask.long().sum(dim=1)
    p_pos = lengths - 1
    bsz = logits.size(0)
    logits_at_p = logits[torch.arange(bsz, device=logits.device), p_pos, :]
    loss = F.cross_entropy(logits_at_p, labels)
    preds = logits_at_p.argmax(dim=-1)
    acc = (preds == labels).float().mean().item()
    return loss, acc


def train_epoch(model, dataloader, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss, acc = _step_at_p(logits, attention_mask, labels)
        loss.backward()
        optimizer.step()

        bsz = input_ids.size(0)
        total_loss += loss.item() * bsz
        total_correct += int(acc * bsz)
        total_count += bsz
    return total_loss / total_count, total_correct / total_count


@torch.no_grad()
def eval_epoch(model, dataloader, device) -> tuple[float, float]:
    model.eval()
    total_loss, total_correct, total_count = 0.0, 0, 0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)
        logits = model(input_ids, attention_mask)
        loss, acc = _step_at_p(logits, attention_mask, labels)
        bsz = input_ids.size(0)
        total_loss += loss.item() * bsz
        total_correct += int(acc * bsz)
        total_count += bsz
    return total_loss / total_count, total_correct / total_count


def main():
    data_dir = Path("/data/young/capstone/graph-learning-benchmarks/submodules/graph-token")
    task = "cycle_check"
    algorithm = "er"

    train_dataset = GraphDataset(data_dir, task, algorithm, "train")
    valid_dataset = GraphDataset(data_dir, task, algorithm, "valid")

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6,
        d_ff=1024,
        dropout=0.1,
    ).to(device)

    print(f"Training on device: {device}")
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}")

    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)

    n_epochs = 10
    best_valid_loss = float("inf")

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)
        valid_loss, valid_acc = eval_epoch(model, valid_loader, device) if len(valid_loader) > 0 else (float("nan"), float("nan"))
        scheduler.step()

        score_to_compare = valid_loss if valid_loss == valid_loss else train_loss
        if score_to_compare < best_valid_loss:
            best_valid_loss = score_to_compare
            torch.save(model.state_dict(), "graph_best_model.pt")

        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        if len(valid_loader) > 0:
            print(f"Valid loss: {valid_loss:.4f} | Valid acc: {valid_acc:.4f}")
        else:
            print("Valid loss: n/a | Valid acc: n/a")
        print("-" * 40)


if __name__ == "__main__":
    main()
