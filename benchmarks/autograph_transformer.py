"""Transformer model for processing graph task data."""

import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR

class GraphDataset(Dataset):
    """Dataset for loading graph task samples with their tokenized representations."""
    
    def __init__(
        self,
        data_dir: str,
        task: str,
        algorithm: str,
        split: str,
        max_seq_length: int = 512,
        max_text_length: int = 128,
        n_samples_per_file: int = -1,
        sampling_seed: int | None = None,
    ):
        self.data_dir = Path(data_dir)
        self.task = task
        self.algorithm = algorithm
        self.split = split
        self.max_seq_length = max_seq_length
        self.max_text_length = max_text_length
        self.n_samples_per_file = n_samples_per_file
        self.sampling_seed = sampling_seed

        self.samples = self._load_samples()
        self.token2idx = {"[PAD]": 0, "[CLS]": 1, "[SEP]": 2}
        self._build_vocabulary()

    def _load_samples(self) -> List[dict]:
        """Load all JSON files for the specified task/algorithm/split."""
        # Handle both cases: data_dir as parent of tasks_autograph or as tasks_autograph itself
        if (self.data_dir / "tasks_autograph").exists():
            task_dir = self.data_dir / "tasks_autograph" / self.task / self.algorithm / self.split
        else:
            task_dir = self.data_dir / self.task / self.algorithm / self.split
        samples = []
        rng = random.Random(self.sampling_seed) if self.sampling_seed is not None else random
        for json_file in sorted(task_dir.glob("*.json")):
            with json_file.open() as f:
                file_samples = json.load(f)
                if isinstance(file_samples, list) and self.n_samples_per_file and self.n_samples_per_file > 0:
                    k = min(len(file_samples), int(self.n_samples_per_file))
                    samples.extend(rng.sample(file_samples, k))
                else:
                    samples.extend(file_samples)

        return samples

    def _build_vocabulary(self):
        """Build token vocabulary from all samples."""
        for sample in self.samples:
            tokens = [str(t) for t in sample["tokens"]]
            for token in tokens:
                if token not in self.token2idx:
                    self.token2idx[token] = len(self.token2idx)

        for sample in self.samples:
            text_tokens = sample["text"].strip().split()
            for tok in text_tokens:
                if tok not in self.token2idx:
                    self.token2idx[tok] = len(self.token2idx)
        
        self.idx2token = {v: k for k, v in self.token2idx.items()}
        self.vocab_size = len(self.token2idx)

    def __len__(self) -> int:
        return len(self.samples)
    
    def _extract_q_p(self, text_tokens: List[str]) -> Tuple[List[str], str]:
        """Return (question_part_tokens_including_<q>_and_<p>, answer_token_after_p)."""
        try:
            i_q = text_tokens.index("<q>")
            i_p = text_tokens.index("<p>")
        except ValueError:
            raise ValueError("Text must contain both <q> and <p> markers.")

        if not (0 <= i_q < i_p < len(text_tokens) - 1):
            raise ValueError("Require <q> ... <p> ANSWER, with at least one token after <p>.")

        question_span = ["<q>"] + text_tokens[i_q+1:i_p] + ["<p>"]
        answer_tok = text_tokens[i_p + 1]
        return question_span, answer_tok

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]

        graph_tokens = [str(t) for t in sample["tokens"]]

        text_tokens = sample["text"].strip().split()

        q_span, answer_tok = self._extract_q_p(text_tokens)
        combined = graph_tokens + q_span

        if len(combined) > self.max_seq_length - 1:
            combined = combined[: self.max_seq_length - 1]

        token_ids = [self.token2idx["[CLS]"]]
        token_ids.extend([self.token2idx.get(t, self.token2idx["[PAD]"]) for t in combined])

        padding_length = self.max_seq_length - len(token_ids)
        if padding_length > 0:
            token_ids.extend([self.token2idx["[PAD]"]] * padding_length)

        attention_mask = [1 if t != self.token2idx["[PAD]"] else 0 for t in token_ids]

        label_id = self.token2idx.get(answer_tok, self.token2idx["[PAD]"])

        return {
            "input_ids": torch.tensor(token_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float),
            "label": torch.tensor(label_id, dtype=torch.long),
            "graph_id": sample["graph_id"],
        }


class TransformerEncoder(nn.Module):
    """Multi-head self-attention transformer encoder block."""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self attention
        attn_output, _ = self.self_attn(x, x, x, key_padding_mask=mask)
        x = x + self.dropout(attn_output)
        x = self.norm1(x)
        
        # Feed forward
        ff_output = self.feed_forward(x)
        x = x + self.dropout(ff_output)
        x = self.norm2(x)
        
        return x

class GraphTransformer(nn.Module):
    """Transformer model for processing graph token sequences."""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 64,
        n_heads: int = 8,
        n_layers: int = 2,
        d_ff: int = 128,
        dropout: float = 0.1,
        max_seq_length: int = 512
    ):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.randn(max_seq_length, d_model))
        
        self.encoder_layers = nn.ModuleList([
            TransformerEncoder(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # input_ids: BS
        
        seq_length = input_ids.size(1)
        
        x = self.token_embedding(input_ids)  # batch_size x seq_length x d_model
        x = x + self.position_embedding[:seq_length].unsqueeze(0)
        x = self.dropout(x)
        
        # Convert attention_mask for transformer (1->False, 0->True for key_padding_mask)
        if attention_mask is not None:
            mask = attention_mask == 0
        else:
            mask = None
        
        # Apply transformer layers
        x = x.transpose(0, 1)  # SBD
        for layer in self.encoder_layers:
            x = layer(x, mask=mask)
        x = x.transpose(0, 1)  # BSD
        
        logits = self.output_projection(x)  # BSV
        
        return logits

def train_epoch(model, dataloader, optimizer, device) -> tuple[float, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_count = 0

    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device)  # (B,)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)  # (B, S, V)

        lengths = attention_mask.long().sum(dim=1)  # (B,)

        bsz = input_ids.size(0)
        logits_at_p = logits[torch.arange(bsz, device=device), p_pos, :]  # (B, V)

        loss = F.cross_entropy(logits_at_p, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * bsz
        preds = logits_at_p.argmax(dim=-1)
        total_correct += (preds == labels).sum().item()
        total_count += bsz

    avg_loss = total_loss / total_count
    acc = total_correct / total_count if total_count > 0 else float('nan')
    return avg_loss, acc


def eval_epoch(model, dataloader, device, task_name: str = "") -> tuple[float, float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_mae = 0.0
    total_count = 0

    with torch.no_grad():
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

            total_loss += loss.item() * bsz
            preds = logits_at_p.argmax(dim=-1)
            total_correct += (preds == labels).sum().item()
            total_count += bsz
            
            # Compute MAE for distance-based tasks like shortest_path
            if "shortest_path" in task_name.lower():
                pred_distances = preds.float()
                label_distances = labels.float()
                total_mae += torch.mean(torch.abs(pred_distances - label_distances)).item() * bsz

    avg_loss = total_loss / total_count if total_count > 0 else float('nan')
    acc = total_correct / total_count if total_count > 0 else float('nan')
    mae = total_mae / total_count if total_count > 0 else 0.0
    return avg_loss, acc, mae


def main():
    data_dir = Path("/data/young/capstone/graph-learning-benchmarks/submodules/graph-token")
    task = "cycle_check"
    algorithm = "er"
    
    train_dataset = GraphDataset(data_dir, task, algorithm, "train")
    valid_dataset = GraphDataset(data_dir, task, algorithm, "valid")

    valid_dataset.token2idx = train_dataset.token2idx
    valid_dataset.idx2token = train_dataset.idx2token
    valid_dataset.vocab_size = train_dataset.vocab_size

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=32)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GraphTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=256,
        n_heads=8,
        n_layers=6
    ).to(device)
    
    print(f"Training on device: {device}")
    print(f"Vocabulary size: {train_dataset.vocab_size}")
    print(f"Number of training samples: {len(train_dataset)}")
    print(f"Number of validation samples: {len(valid_dataset)}")
    
    optimizer = Adam(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=10)
    
    n_epochs = 10
    best_valid_loss = float('inf')
    
    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, device)

        valid_loss, valid_acc = eval_epoch(model, valid_loader, device)

        scheduler.step()

        score_to_compare = valid_loss
        if score_to_compare < best_valid_loss:
            best_valid_loss = score_to_compare
            torch.save(model.state_dict(), "autograph_best_model.pt")

        print(f"Epoch {epoch+1}/{n_epochs}")
        print(f"Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f}")
        print(f"Valid loss: {valid_loss:.4f} | Valid acc: {valid_acc:.4f}")
        print("-" * 40)

if __name__ == "__main__":
    main()
