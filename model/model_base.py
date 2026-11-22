import torch
import math
import torch.nn as nn
from transformers import AutoModel, AutoConfig


class DMGModel(nn.Module):
    def __init__(self, args, lab_classes, rel_classes, device):
        super(DMGModel, self).__init__()
        self.args = args
        self.config = AutoConfig.from_pretrained(args.bert_path)
        self.encoder = AutoModel.from_pretrained(args.bert_path)
        if args.fix_bert_embedding:
            self.encoder.embeddings.word_embeddings.weight.requires_grad = False
            self.encoder.embeddings.position_embeddings.weight.requires_grad = False
            self.encoder.embeddings.token_type_embeddings.weight.requires_grad = False
        self.dropout = nn.Dropout(p=args.drop_rate)
        self.ent_embed = nn.Embedding(num_embeddings=1000, embedding_dim=self.args.ent_dim)
        self.edge_embed = nn.Embedding(1000, args.edge_dim)
        self.BiGRU = nn.GRU(input_size=self.config.hidden_size + self.args.ent_dim,
                            hidden_size=self.config.hidden_size // 2,
                            bidirectional=True)
        self.LayerNorm = nn.LayerNorm(normalized_shape=self.config.hidden_size)

        self.tri_classifier = nn.Linear(self.config.hidden_size, lab_classes)
        self.rel_classifier = nn.Linear(self.config.hidden_size, rel_classes)
        self.lab_classifier = nn.Linear(self.config.hidden_size, 2)

    def reset_parameters(self):
        for i in self.named_parameters():
            if len(i[1].size()) == 1:
                std = 1.0 / math.sqrt(i[1].size(0))
                nn.init.uniform_(i[1], -std, std)
            else:
                nn.init.xavier_normal_(i[1])

    def combine(self, token_encode, seq_lens):
        # 加入BiLSTM获得上下文编码
        token_encode_pack = torch.nn.utils.rnn.pack_padded_sequence(token_encode, seq_lens.cpu(), batch_first=True,
                                                                    enforce_sorted=False)
        gru_encode, _ = self.BiGRU(token_encode_pack)
        gru_encode_unpack, _ = torch.nn.utils.rnn.pad_packed_sequence(gru_encode, batch_first=True)
        return gru_encode_unpack

    def forward(self, input_ids, attention_mask, offsets, ent_labs, seq_lens, graphs):
        # last_hidden_state, _ = self.encoder(input_ids, attention_mask)
        last_hidden_state = self.encoder(input_ids, attention_mask).last_hidden_state
        batch_size = last_hidden_state.shape[0]
        range_vector = torch.LongTensor(batch_size).to(last_hidden_state.device).fill_(1).cumsum(0) - 1
        bert_encode = last_hidden_state[range_vector.unsqueeze(1), offsets]
        encode = self.dropout(bert_encode)
        ent_embed = self.ent_embed(ent_labs)

        token_encode = torch.cat([encode, ent_embed], dim=-1)
        gru_encode = self.combine(token_encode=token_encode, seq_lens=seq_lens)
        encode = self.LayerNorm(gru_encode + encode)
        t_logits = self.tri_classifier(encode)
        r_logits = self.rel_classifier(encode)
        s_logits = self.lab_classifier(last_hidden_state[:, 0, :])
        return {"tri": t_logits, "rel": r_logits, "lab": s_logits}

    def high_order(self, graph):
        res = graph
        edge_dim = res.shape[-1]
        res = torch.stack([torch.matmul(res[:, :, :, j], graph[:, :, :, j]) for j in range(edge_dim)], dim=-1)
        return res
