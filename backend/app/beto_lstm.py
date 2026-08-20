"""BETO encoder with a bidirectional LSTM classification head."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn
from transformers.modeling_outputs import SequenceClassifierOutput


class BETOBiLSTM(nn.Module):
    """Apply BiLSTM sequence modeling to BETO token representations."""

    def __init__(
        self,
        encoder: nn.Module,
        hidden_size: int = 256,
        num_layers: int = 1,
        dropout: float = 0.3,
        num_labels: int = 2,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        encoder_hidden_size = int(encoder.config.hidden_size)
        self.lstm = nn.LSTM(
            input_size=encoder_hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size * 2, num_labels)

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor | None = None,
        labels: Tensor | None = None,
    ) -> SequenceClassifierOutput:
        """Return logits and optional cross-entropy loss."""
        encoder_inputs: dict[str, Any] = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            encoder_inputs["token_type_ids"] = token_type_ids
        sequence = self.encoder(**encoder_inputs).last_hidden_state
        lengths = attention_mask.sum(dim=1).to(torch.int64).cpu().clamp_min(1)
        packed = nn.utils.rnn.pack_padded_sequence(
            sequence, lengths, batch_first=True, enforce_sorted=False
        )
        _, (_, hidden) = self.lstm(packed)
        pooled = torch.cat((hidden[-2], hidden[-1]), dim=1)
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = nn.functional.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)
