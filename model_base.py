import torch
import math
import torch.nn as nn
from transformers import AutoModel, AutoConfig

class EventGraphReason(nn.Module):
    def __init__(self, using_reason, inner_dim):
        super(EventGraphReason, self).__init__()
        self.using_reason = using_reason
        self.inner_dim = inner_dim
        self.dense = nn.Linear(())
    def forward(self, context_outputs, token_mask, logits):
        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # TODO:修改为Linear获取？

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = token_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5


class MentionPointer(nn.Module):
    def __init__(self, args, ent_type_size, inner_dim, device, RoPE=True):
        super().__init__()
        self.args = args
        self.device = device
        self.config = AutoConfig.from_pretrained(args.bert_path)
        self.encoder = AutoModel.from_pretrained(args.bert_path)
        if args.fix_bert_embedding:
            self.encoder.embeddings.word_embeddings.weight.requires_grad = False
            self.encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.encoder.embeddings.token_type_embeddings.weight.requires_grad = False
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.ent_embed = nn.Embedding(num_embeddings=1000, embedding_dim=self.args.ent_dim)
        self.BiGRU = nn.GRU(input_size=self.config.hidden_size + self.args.ent_dim,
                            hidden_size=self.config.hidden_size // 2,
                            bidirectional=True)
        self.LayerNorm = nn.LayerNorm(normalized_shape=self.config.hidden_size)

        self.ent_type_size = ent_type_size
        self.inner_dim = inner_dim
        self.hidden_size = self.config.hidden_size
        self.dense = nn.Linear(self.hidden_size, self.ent_type_size * self.inner_dim * 2)
        self.RoPE = RoPE

    def sinusoidal_position_embedding(self, batch_size, seq_len, output_dim):
        position_ids = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(-1)

        indices = torch.arange(0, output_dim // 2, dtype=torch.float)
        indices = torch.pow(10000, -2 * indices / output_dim)
        embeddings = position_ids * indices
        embeddings = torch.stack([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)
        embeddings = embeddings.repeat((batch_size, *([1] * len(embeddings.shape))))
        embeddings = torch.reshape(embeddings, (batch_size, seq_len, output_dim))
        embeddings = embeddings.to(self.device)
        return embeddings

    def combine(self, token_encode, seq_lens):
        # 加入BiLSTM获得上下文编码
        token_encode_pack = torch.nn.utils.rnn.pack_padded_sequence(token_encode, seq_lens.cpu(), batch_first=True,
                                                                    enforce_sorted=False)
        gru_encode, _ = self.BiGRU(token_encode_pack)
        gru_encode_unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_encode, batch_first=True)
        return gru_encode_unpack

    def forward(self, input_ids, offsets, ent_labs, seq_lens):
        attention_mask = (input_ids > 0).long()
        token_mask = (offsets > 0).long()
        # last_hidden_state, _ = self.encoder(input_ids, attention_mask)
        last_hidden_state = self.encoder(input_ids, attention_mask).last_hidden_state
        batch_size = last_hidden_state.shape[0]
        aaa = torch.max(offsets)
        range_vector = torch.LongTensor(batch_size).to(last_hidden_state.device).fill_(1).cumsum(0) - 1
        bert_encode = last_hidden_state[range_vector.unsqueeze(1), offsets]
        encode = self.dropout(bert_encode)
        ent_embed = self.ent_embed(ent_labs)

        token_encode = torch.cat([encode, ent_embed], dim=-1)
        gru_encode = self.combine(token_encode=token_encode, seq_lens=seq_lens)
        context_outputs = self.LayerNorm(gru_encode + encode)

        # last_hidden_state:(batch_size, seq_len, hidden_size)
        last_hidden_state = context_outputs

        batch_size = last_hidden_state.size()[0]
        seq_len = last_hidden_state.size()[1]

        # outputs:(batch_size, seq_len, ent_type_size*inner_dim*2)
        outputs = self.dense(last_hidden_state)
        outputs = torch.split(outputs, self.inner_dim * 2, dim=-1)
        # outputs:(batch_size, seq_len, ent_type_size, inner_dim*2)
        outputs = torch.stack(outputs, dim=-2)
        # qw,kw:(batch_size, seq_len, ent_type_size, inner_dim)
        qw, kw = outputs[..., :self.inner_dim], outputs[..., self.inner_dim:]  # TODO:修改为Linear获取？

        if self.RoPE:
            # pos_emb:(batch_size, seq_len, inner_dim)
            pos_emb = self.sinusoidal_position_embedding(batch_size, seq_len, self.inner_dim)
            # cos_pos,sin_pos: (batch_size, seq_len, 1, inner_dim)
            cos_pos = pos_emb[..., None, 1::2].repeat_interleave(2, dim=-1)
            sin_pos = pos_emb[..., None, ::2].repeat_interleave(2, dim=-1)
            qw2 = torch.stack([-qw[..., 1::2], qw[..., ::2]], -1)
            qw2 = qw2.reshape(qw.shape)
            qw = qw * cos_pos + qw2 * sin_pos
            kw2 = torch.stack([-kw[..., 1::2], kw[..., ::2]], -1)
            kw2 = kw2.reshape(kw.shape)
            kw = kw * cos_pos + kw2 * sin_pos

        # logits:(batch_size, ent_type_size, seq_len, seq_len)
        logits = torch.einsum('bmhd,bnhd->bhmn', qw, kw)

        # padding mask
        pad_mask = token_mask.unsqueeze(1).unsqueeze(1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask_h = attention_mask.unsqueeze(1).unsqueeze(-1).expand(batch_size, self.ent_type_size, seq_len, seq_len)
        # pad_mask = pad_mask_v&pad_mask_h
        logits = logits * pad_mask - (1 - pad_mask) * 1e12

        # 排除下三角
        mask = torch.tril(torch.ones_like(logits), -1)
        logits = logits - mask * 1e12

        return logits / self.inner_dim ** 0.5


    def multilabel_categorical_crossentropy(self, y_pred, y_true):
        """
        https://kexue.fm/archives/7359
        """
        y_pred = (1 - 2 * y_true) * y_pred  # -1 -> pos classes, 1 -> neg classes
        y_pred_neg = y_pred - y_true * 1e12  # mask the pred outputs of pos classes
        y_pred_pos = (y_pred - (1 - y_true) * 1e12)  # mask the pred outputs of neg classes
        zeros = torch.zeros_like(y_pred[..., :1])
        y_pred_neg = torch.cat([y_pred_neg, zeros], dim=-1)
        y_pred_pos = torch.cat([y_pred_pos, zeros], dim=-1)
        neg_loss = torch.logsumexp(y_pred_neg, dim=-1)
        pos_loss = torch.logsumexp(y_pred_pos, dim=-1)

        return (neg_loss + pos_loss).mean()

    def loss_fun(self, y_true, y_pred):
        """
        y_true:(batch_size, ent_type_size, seq_len, seq_len)
        y_pred:(batch_size, ent_type_size, seq_len, seq_len)
        """
        batch_size, ent_type_size = y_pred.shape[:2]
        y_true = y_true.reshape(batch_size * ent_type_size, -1)
        y_pred = y_pred.reshape(batch_size * ent_type_size, -1)
        loss = self.multilabel_categorical_crossentropy(y_true, y_pred)
        return loss
