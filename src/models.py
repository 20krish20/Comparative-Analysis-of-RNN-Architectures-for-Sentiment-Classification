from typing import Literal, Tuple

import torch
import torch.nn as nn


ArchitectureType = Literal["rnn", "lstm", "bilstm"]
ActivationType = Literal["relu", "tanh", "sigmoid"]


def get_activation(name: ActivationType) -> nn.Module:
    """
    Map a string to a PyTorch activation module for the MLP head.

    Note: The final output layer ALWAYS uses Sigmoid for binary classification.
    """
    name = name.lower()
    if name == "relu":
        return nn.ReLU()
    elif name == "tanh":
        return nn.Tanh()
    elif name == "sigmoid":
        return nn.Sigmoid()
    else:
        raise ValueError(f"Unsupported activation: {name}")


class TextRNNClassifier(nn.Module):
    """
    Text classifier with:

        Embedding -> RNN/LSTM/BiLSTM -> [Dropout] -> 2-layer MLP head -> Sigmoid

    This model is designed to satisfy your project spec:

        - Embedding dim = 100
        - 2 hidden recurrent layers, hidden size = 64
        - Dropout ~ 0.3–0.5
        - Fully connected output layer with sigmoid for binary classification
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 100,
        architecture: ArchitectureType = "lstm",
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.5,
        activation: ActivationType = "relu",
        pad_idx: int = 0,
    ):
        """
        Parameters
        ----------
        vocab_size : int
            Size of vocabulary (including PAD + OOV).
        embedding_dim : int
            Dimension of word embeddings (fixed to 100 by spec, but kept configurable).
        architecture : {"rnn", "lstm", "bilstm"}
            Recurrent architecture type.
        hidden_size : int
            Size of the hidden state in each recurrent layer.
        num_layers : int
            Number of recurrent layers (spec: 2).
        dropout : float
            Dropout probability applied between layers and in the MLP head.
        activation : {"relu", "tanh", "sigmoid"}
            Activation function used in the MLP head (not the final output).
        pad_idx : int
            Index used for padding tokens (so we can tell embedding to ignore it).
        """
        super().__init__()

        self.architecture = architecture.lower()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = self.architecture == "bilstm"

        # 1) Embedding layer
        self.embedding = nn.Embedding(
            num_embeddings=vocab_size,
            embedding_dim=embedding_dim,
            padding_idx=pad_idx,
        )

        # 2) Recurrent encoder
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.architecture == "rnn":
            # Simple RNN (we use tanh nonlinearity inside; head activation varies separately)
            self.rnn = nn.RNN(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                nonlinearity="tanh",  # internal; we vary head activation separately
                bidirectional=False,
            )
            self.is_lstm = False

        elif self.architecture in {"lstm", "bilstm"}:
            self.rnn = nn.LSTM(
                input_size=embedding_dim,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                dropout=rnn_dropout,
                bidirectional=self.bidirectional,
            )
            self.is_lstm = True

        else:
            raise ValueError(f"Unsupported architecture type: {architecture}")

        # Dimensionality of the encoder output vector going into the MLP head
        num_directions = 2 if self.bidirectional else 1
        encoder_out_dim = hidden_size * num_directions

        # 3) MLP head: encoder_out_dim -> 64 -> 1
        self.head_dropout1 = nn.Dropout(dropout)
        self.fc1 = nn.Linear(encoder_out_dim, 64)
        self.head_activation = get_activation(activation)
        self.head_dropout2 = nn.Dropout(dropout)
        self.fc_out = nn.Linear(64, 1)

        # 4) Output Sigmoid for binary classification
        self.out_act = nn.Sigmoid()

        # Optional: initialize weights (basic scheme)
        self._init_weights()

    # ------------------------------------------------------------------
    # Forward helpers
    # ------------------------------------------------------------------

    def _init_weights(self):
        # Small-scale initialization for fc layers (helps stability a bit)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def _get_lstm_last_hidden(self, rnn_output) -> torch.Tensor:
        """
        Extract the final hidden representation from LSTM/RNN outputs.

        For LSTM: rnn_output is (output, (h_n, c_n))
        For RNN:  rnn_output is (output, h_n)
        """
        if self.is_lstm:
            output, (h_n, c_n) = rnn_output
        else:
            output, h_n = rnn_output

        # h_n shape: (num_layers * num_directions, batch, hidden_size)
        # We take the last layer's hidden state for each direction and concat.
        if self.bidirectional:
            # Last layer forward ([-2]) and backward ([-1])
            h_forward = h_n[-2, :, :]  # (batch, hidden_size)
            h_backward = h_n[-1, :, :]  # (batch, hidden_size)
            h_final = torch.cat([h_forward, h_backward], dim=1)  # (batch, 2*hidden_size)
        else:
            h_final = h_n[-1, :, :]  # last layer, (batch, hidden_size)

        return h_final  # (batch, encoder_out_dim)

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Parameters
        ----------
        x : LongTensor of shape (batch_size, seq_len)
            Batch of token IDs.

        Returns
        -------
        probs : Tensor of shape (batch_size,)
            Predicted probability of the positive class.
        """
        # 1) Look up embeddings
        # x: (batch, seq_len) -> (batch, seq_len, embedding_dim)
        emb = self.embedding(x)

        # 2) Run through RNN/LSTM
        # With batch_first=True, input is (batch, seq_len, embedding_dim)
        rnn_out = self.rnn(emb)
        h_final = self._get_lstm_last_hidden(rnn_out)  # (batch, encoder_out_dim)

        # 3) MLP head
        z = self.head_dropout1(h_final)
        z = self.fc1(z)
        z = self.head_activation(z)
        z = self.head_dropout2(z)
        logits = self.fc_out(z)  # (batch, 1)

        # 4) Sigmoid output -> (batch,)
        probs = self.out_act(logits).squeeze(1)
        return probs
