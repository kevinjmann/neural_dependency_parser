import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SelfAttention(nn.Module):
    def __init__(self, embed_size, heads):
        super().__init__()
        self.embed_size = embed_size
        self.heads = heads
        self.head_dim = embed_size // heads
        assert (self.head_dim * self.heads == embed_size), "Embed size needs to be divisible by heads"

        # these attn heads take only a subset of an input
        self.W_v = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_k = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.W_q = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.fc_out = nn.Linear(heads*self.head_dim, embed_size)

    def forward(self, values, keys, query, mask):
        # batch size
        N = query.shape[0]
        value_len, key_len, query_len = values.shape[1], keys.shape[1], query.shape[1]

        #split embedding into self.heads pieces
        values = values.reshape(N, value_len, self.heads, self.head_dim)
        keys = keys.reshape(N, key_len, self.heads, self.head_dim)
        queries = query.reshape(N, query_len, self.heads, self.head_dim)

        values = self.W_v(values)
        keys = self.W_k(keys)
        queries = self.W_q(queries)
        # multiply keys and queries Attn = softmax(Q*K_T/sqrt(head_dim)) * V
        # queries shape: (N, query_len, heads, head_dim)
        # keys shape: (N, key_len, heads, head_dim)
        # energy shape: (N, heads, query_len, key_len)
        energy = torch.einsum("nqhd,nkhd->nhqk", [queries, keys])

        if mask is not None:
            energy = energy.masked_fill(mask == 0, float("-1e28"))

        attention = torch.softmax(energy/(self.embed_size ** (1/2)), dim=3)
        # attention shape: (N, heads, query_len, key_len)
        # values shape: (N, value_len, self.heads, self.head_dim)
        # desired N, query_len, heads, head_dim then concatenate
        out = torch.einsum('nhql,nlhd->nlhd', [attention, values]).reshape(
            N, query_len, self.heads * self.head_dim
        )

        out = self.fc_out(out)
        return out


class TransformerBlock(nn.Module):
    def __init__(self, embed_size, heads, dropout, forward_expansion):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm1 = nn.LayerNorm(embed_size)
        self.norm2 = nn.LayerNorm(embed_size)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_size, forward_expansion * embed_size),
            nn.ReLU(),
            nn.Linear(forward_expansion * embed_size, embed_size)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, value, key, query, mask):
        attention = self.attention(value, key, query, mask)
        x = self.dropout(self.norm1(attention + query))
        forward = self.feed_forward(x)
        out = self.dropout(self.norm2(forward + x))
        return out


class Encoder(nn.Module):
    def __init__(
        self,
        src_vocab_size,
        embed_size,
        num_layers,
        heads,
        device,
        forward_expansion,
        dropout,
        max_length,
        num_lang_features,
    ):
        super().__init__()
        self.embed_size = embed_size
        self.device = device
        self.word_embedding = nn.Embedding(src_vocab_size, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)
        self.language_embedding = nn.Linear(num_lang_features, embed_size)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    embed_size, heads, dropout, forward_expansion
                )
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask, language_vector):
        N, seq_length = x.shape
        positions = torch.arange(0, seq_length).expand(N, seq_length).to(self.device)
        out = self.dropout((self.word_embedding(x) + self.position_embedding(positions) + self.language_embedding(language_vector)))
        for layer in self.layers:
            out = layer(out, out, out, mask)

        return out


class DecoderBlock(nn.Module):
    def __init__(self, embed_size, heads, forward_expansion, dropout, device):
        super().__init__()
        self.attention = SelfAttention(embed_size, heads)
        self.norm = nn.LayerNorm(embed_size)
        self.transformer_block = TransformerBlock(
            embed_size, heads, dropout, forward_expansion
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, value, key, src_mask, target_mask):
        attention = self.attention(x, x, x, target_mask)
        query = self.dropout(
            self.norm(
                attention + x
            )
        )
        out = self.transformer_block(value, key, query, src_mask)
        return out


class Decoder(nn.Module):
    def __init__(
        self,
        trg_vocab_size_sr,
        trg_vocab_size_deprel,
        embed_size,
        num_layers,
        heads,
        forward_expansion,
        dropout,
        device,
        max_length
    ):
        super().__init__()

        self.sr_embedding = nn.Embedding(trg_vocab_size_sr, embed_size)
        self.deprel_embedding = nn.Embedding(trg_vocab_size_deprel, embed_size)
        self.position_embedding = nn.Embedding(max_length, embed_size)

        self.layers = nn.ModuleList([
            DecoderBlock(embed_size, heads, forward_expansion, dropout, device)
            for _ in range(num_layers)
        ])

        self.fc_out_sr = nn.Linear(embed_size, trg_vocab_size_sr)
        self.fc_out_deprel = nn.Linear(embed_size, trg_vocab_size_deprel)
        self.dropout = nn.Dropout(dropout)

    # x input to decoder
    def forward(self, x_sr, x_deprel, enc_out, src_mask, target_mask):
        N, seq_len = x_sr.shape
        positions = torch.arange(0, seq_len).expand(N, seq_len).to(self.device)
        x = self.dropout((self.sr_embedding(x_sr) + self.deprel_embedding(x_deprel) + self.position_embedding(positions)))

        for layer in self.layers:
            x = layer(x, enc_out, enc_out, src_mask, target_mask)

        out_sr = self.fc_out_sr(x)
        out_deprel = self.fc_out_deprel(x)
        return out_sr, out_deprel


class DependencyParserTransformer(nn.Module):
    """
    Slightly modified version of the transformer architecture for use in a shift-reduce (SR) dependency parser.
    Maintains two input and output vocabularies corresponding to:
    input 1: universal dependency treebank UPOS tags
    input 2: a vector containing 1 or 0 corresponding to features of the input language originally taken from WALS
    output 1: Shift Reduce dependency parser actions
    output 2: Dependency labels
    """
    def __init__(
        self,
        src_vocab_size,
        target_vocab_sr_size,
        target_vocab_deprel_size,
        src_pad_idx,
        trg_pad_idx,
        embed_size=256,
        num_layers=6,
        forward_expansion=4,
        heads=8,
        dropout=0,
        device='cuda',
        max_length=100,
        num_lang_features=20,
    ):
        super().__init__()
        self.encoder = Encoder(
            src_vocab_size,
            embed_size,
            num_layers,
            heads,
            device,
            forward_expansion,
            dropout,
            max_length,
            num_lang_features)
        self.decoder = Decoder(
            target_vocab_sr_size,
            target_vocab_deprel_size,
            embed_size,
            num_layers,
            heads,
            forward_expansion,
            dropout,
            device,
            max_length,
        )

        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.device = device

    def make_src_mask(self, src):
        src_mask = (src != self.src_pad_idx).unsqueeze(1).unsqueeze(2)
        # N, 1, 1, src_length
        return src_mask.to(self.device)

    def make_trg_mask(self, trg):
        N, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(
            N, 1, trg_len, trg_len
        )
        return trg_mask.to(self.device)

    def forward(self, src, lang_vector, trg_sr, trg_deprel):
        src_mask = self.make_src_mask(src)
        trg_mask = self.make_trg_mask(trg_sr)
        enc_src = self.encoder(src, src_mask, lang_vector)
        out_sr, out_deprel = self.decoder(trg_sr, trg_deprel, enc_src, src_mask, trg_mask)
        return out_sr, out_deprel


def multi_label_loss(preds_sr, preds_deprel, yb_sr, yb_deprel, trg_pad_idx):
    loss = F.cross_entropy(preds_sr, yb_sr, ignore_index=trg_pad_idx, reduction='sum') + F.cross_entropy(preds_deprel, yb_deprel, ignore_index=trg_pad_idx, reduction='sum')
    return loss


def accuracy(outputs_sr, outputs_deprel, labels_sr, labels_deprel):
    _, preds_sr = torch.max(outputs_sr, dim=1)
    _, preds_deprel = torch.max(outputs_deprel, dim=1)
    acc_sr = torch.tensor(torch.sum(preds_sr == labels_sr).item() / len(outputs_sr))
    acc_deprel = torch.tensor(torch.sum(preds_deprel == labels_deprel).item() / len(outputs_deprel))
    return (acc_sr + acc_deprel) / 2


def loss_batch(model, loss_fn, xb, xb_lang, yb_sr, yb_deprel, trg_pad_idx, opt=None, metric=None):
    preds_sr, preds_deprel = model(xb, xb_lang, yb_sr, yb_deprel)
    loss = loss_fn(preds_sr, preds_deprel, yb_sr, yb_deprel, trg_pad_idx)
    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()
    metric_value = None
    if metric is not None:
        metric_value = metric(preds_sr, preds_deprel, yb_sr, yb_deprel)
    return loss, len(xb), metric_value


def evaluate(model, loss_fn, valid_dl, trg_pad_idx, metric=None):
    with torch.no_grad():
        results = [loss_batch(model, loss_fn, xb, yb_sr, yb_deprel, trg_pad_idx, metric=metric) for xb, yb_sr, yb_deprel in valid_dl]
        loss, nums, metrics = zip(*results)
        total = np.sum(nums)
        avg_loss = np.sum(np.multiply(loss, nums)) / total
        avg_metric = None
        if metric is not None:
            avg_metric = np.sum(np.multiply(metrics, nums)) / total
        return avg_loss, nums, avg_metric


def fit(epochs, model, loss_fn, train_dl, valid_dl, opt_fn=None, metric=None):
    train_losses, val_losses, val_metrics = [],[],[]
    if opt_fn is None:
        opt_fn = torch.optim.Adam
    opt = opt_fn(model.parameters(), lr=0.005)
    for epoch in epochs:
        model.train()
        for xb, xb_lang, yb_sr, yb_deprel in train_dl:
            loss, _, _ = loss_batch(model, loss_fn, xb, xb_lang, yb_sr, yb_deprel, opt, metric)
        model.eval()
        val_loss, _, val_metric = evaluate(model, loss_fn, valid_dl, metric)
        val_losses.append(val_loss)
        val_metrics.append(val_metric)
        train_losses.append(loss)
        if metric is None:
            print(f'Epoch {epoch + 1}/{epochs} val_loss {val_loss}')
        else:
            print(f'Epoch {epoch+1}/{epochs} val_loss {val_loss} {metric.__name__} {val_metric}')
        return train_losses, val_losses, val_metrics
