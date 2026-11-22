import json
import os
import re
import torch
import math
import random
import pickle
import numpy as np
from collections import Counter
from tqdm import tqdm

Theme_events = ["Gene_expression", "Transcription", "Localization", "Protein_catabolism"]
Theme_Cause_events = ["Regulation", "Negative_regulation", "Positive_regulation", "Phosphorylation"]
Other_events = ["Protein_modification", "Ubiquitination", "Deacetylation", "Acetylation", "Entity", "Anaphora"]
MThemes_events = ["Binding"]
all_events = Theme_events + Theme_Cause_events + MThemes_events + Other_events

class SpanDataloader:
    def __init__(self,
                 train_path,
                 devel_path,
                 test_path,
                 task_name,
                 max_len=512,
                 tokenizer=None,
                 logger=None, ):
        self.train_path = train_path
        self.devel_path = devel_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.logger = logger
        self.task_name = task_name
        self.max_len = max_len

    def load_data(self, file_path, dataset_name):
        lines = pickle.load(open(file_path, mode="rb"))
        return lines

    def build_map(self):
        if not os.path.exists(f"experiments/{self.task_name}/schema.json"):
            data = self.load_data(file_path=self.train_path, dataset_name="train")
            data += self.load_data(file_path=self.devel_path, dataset_name="devel")
            deps, tags, trigger, entity, rels = [], [], [], [], []
            for instance in tqdm(data, desc="build schema"):
                deps.extend([item[1] for item in instance["token"]["mask_dep"]])
                tags.extend(instance["token"]["mask_tag"])
                tris = instance["trigger"]
                ents = instance["entity"]
                relations = instance["relation"]
                for tri in tris.values():
                    trigger.append(tri["type"])
                for ent in ents.values():
                    entity.append(ent["type"])
                for relation in relations:
                    rels.append(relation[0])

            dep_tuple = ["O"] + [item[0] for item in
                                 sorted(dict(Counter(deps)).items(), key=lambda x: x[1], reverse=True)] + ["self-loop"]
            tag_tuple = ["O"] + [item[0] for item in
                                 sorted(dict(Counter(tags)).items(), key=lambda x: x[1], reverse=True)]
            tri_tuple = [item[0] for item in
                         sorted(dict(Counter(trigger)).items(), key=lambda x: x[1], reverse=True)]
            rel_tuple = [item[0] for item in
                         sorted(dict(Counter(rels)).items(), key=lambda x: x[1], reverse=True)]

            ent_tuple = ["O"]
            for item in [item[0] for item in sorted(dict(Counter(entity)).items(), key=lambda x: x[1], reverse=True)]:
                ent_tuple.append(f"B-{item}")
                ent_tuple.append(f"I-{item}")

            dep2id = {key: i for i, key in enumerate(dep_tuple)}
            tag2id = {key: i for i, key in enumerate(tag_tuple)}
            tri2id = {key: i for i, key in enumerate(tri_tuple)}
            ent2id = {key: i for i, key in enumerate(ent_tuple)}
            rel2id = {key: i for i, key in enumerate(rel_tuple)}

            schema = {"lab": tri2id, "ent": ent2id, "dep": dep2id, "tag": tag2id, "rel": rel2id}
            with open(f"experiments/{self.task_name}/schema.json", encoding="utf-8", mode="w") as f:
                json.dump(schema, f, ensure_ascii=False, indent=4)
        else:
            schema = json.load(open(f"experiments/{self.task_name}/schema.json", encoding="utf-8", mode="r"))
        return schema

    def get_tensor(self, data_path, dataset_name, schema):
        all_data = self.load_data(file_path=data_path, dataset_name="train")
        ent2id, lab2id, tag2id, dep2id = schema["ent"], schema["lab"], schema["tag"], schema["dep"]
        rel2id = schema["rel"]
        pp_data, trigger_count, entity_count, total_roles, contain_roles, pos, neg = [], 0, 0, 0, 0, 0, 0
        e = set()
        total_spans, remove_spans = 0, 0
        for i in tqdm(range(len(all_data)), desc=f"generating {dataset_name} instances"):
            instance = all_data[i]
            sent_id = instance["id"]
            doc_id = sent_id.split(":s")[0]
            tokens = [item[3] for item in instance["token"]["mask_token"]]
            char_map = [[item[0], item[1]] for item in instance["token"]["mask_token"]]
            tris = instance["trigger"]
            ents = instance["entity"]
            relations = instance["relation"]
            entities_str = []
            ent_lab = ["O"] * len(tokens)
            trigger_count += len(tris)
            entity_count += len(ents)
            for key, ent in ents.items():
                text = ent["text"]
                type_ = ent["type"]
                char_start = ent["doc_start"]
                char_end = ent["doc_end"]
                mask_start = ent["mask_start"]
                mask_end = ent["mask_end"]
                ent_str = f"{doc_id}.{key}&&{char_start}-{char_end}:{text}>>{type_}"
                if ent_str not in entities_str:
                    entities_str.append(ent_str)
                    e.add(ent_str)
                for index in range(mask_start, mask_end):
                    if index == mask_start:
                        ent_lab[index] = f"B-{type_}"
                    else:
                        ent_lab[index] = f"I-{type_}"

            ent_lab = [ent2id[item] for item in ent_lab]
            org_txt = instance["text"]
            sent_start = int(instance["offset"].split("-")[0])

            token_sent = [self.tokenizer.cls_token]
            bert_span = []
            offsets = []
            bert_start, bert_end = 1, 0
            for token_index, token in enumerate(tokens):
                bert_tk = self.tokenizer.tokenize(token)
                bert_end = bert_start + len(bert_tk)
                token_sent.extend(bert_tk)
                bert_span.append((bert_start, bert_end))
                offsets.append(bert_start)
                bert_start = bert_end

            token_sent.append(self.tokenizer.sep_token)

            bert_tk_id = self.tokenizer.encode(' '.join(tokens))

            assert len(token_sent) == len(bert_tk_id)
            assert len(tokens) == len(offsets)

            if len(bert_tk_id) > self.max_len:
                # 截取offset
                offsets = [idx for idx in offsets if idx < self.max_len]
                bert_tk_id = bert_tk_id[:self.max_len]
                ent_lab = ent_lab[:len(offsets)]
                tokens = tokens[:len(offsets)]
            token_mx_len = len(offsets)

            mention_spans = []
            for relation in relations:
                total_spans += 1
                if relation[1] < token_mx_len and relation[2] <= token_mx_len:
                    mention_spans.append((relation[1], relation[2], rel2id[relation[0]], relation[0]))
                else:
                    remove_spans += 1
            assert len(relations) == sum([len(item) for item in instance["gold"].values()])
            pp_data.append(
                {
                    "sent_id": sent_id,
                    "doc_id": doc_id,
                    "text": tokens,
                    "sent_start": sent_start,
                    "org_txt": org_txt,
                    "bert":
                        {"tokens": tokens, "bert-tokens": token_sent, "input_ids": bert_tk_id, "offsets": offsets},
                    "entity": entities_str,
                    "ent_lab": ent_lab,
                    "mention": mention_spans,
                    "char_map": char_map,
                    "gold": instance["gold"]
                }
            )
        info = [f"entity count: {len(e)}", f"total mentions: {total_spans}", f"remove mentions: {remove_spans}"]
        return pp_data, info


import os
import json
import torch
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, data):
        self.data = data
        self.length = len(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def my_collate(batch, lab_len):
    max_bert_length = max([len(item['bert']['input_ids']) for item in batch])
    max_token_length = max([len(item['bert']['offsets']) for item in batch])
    input_ids, offsets, ent_labs, seq_lens, labs = [], [], [], [], []
    info = {"sent_idx": [], "org_text": {}, "token": {},"char_map": {}, "entity": {}, "sent_start": {}, "trigger": {},
            "argument": {}, "network": {}, "mention": {}}
    for k, instance in enumerate(batch):
        bert_info = instance['bert']
        input_id = bert_info['input_ids']
        offset = bert_info['offsets']
        ent_lab = instance["ent_lab"]
        mentions = instance["mention"]
        entity_strs = instance["entity"]
        if len(input_id) <= max_bert_length:  # Bert补全
            pad_token = [0] * (max_bert_length - len(input_id))
            input_ids.append(input_id + pad_token)
        if len(offset) <= max_token_length:
            pad_token = [0] * (max_token_length - len(offset))
            offsets.append(offset + pad_token)
            ent_labs.append(ent_lab + pad_token)
            assert len(ent_labs[-1]) <= max_token_length
            seq_lens.append(len(offset))
            sentence_id = instance["sent_id"]
            info["char_map"][sentence_id] = instance["char_map"]
            info["sent_start"][sentence_id] = instance["sent_start"]
            info["org_text"][sentence_id] = instance["org_txt"]
            info["trigger"][sentence_id] = instance["gold"]["trigger"]
            info["argument"][sentence_id] = instance["gold"]["argument"]
            info["network"][sentence_id] = instance["gold"]["network"]
            info["token"][sentence_id] = instance["text"]
            info["entity"][sentence_id] = set()

            for entity_str in entity_strs:
                doc_id, rest = entity_str.split("&&")
                span, rest = rest.split(":")
                span_txt, type_ = rest.split(">>")
                start, end = int(span.split("-")[0]), int(span.split("-")[1])
                info["entity"][sentence_id].add((doc_id, type_, start, end, span_txt))

            lab = np.zeros(shape=[lab_len, max_token_length, max_token_length],
                           dtype=np.int64)  # 因为头位置不一定大于尾位置，所以会有下三角矩阵的内容，所以要转化过来才行
            info["mention"][sentence_id] = set()
            for mention in mentions:
                if mention[0] < max_token_length and mention[1] <= max_token_length:
                    if mention[-1] in all_events:
                        lab[mention[2]][mention[0]][mention[1] - 1] = 1
                    else:
                        lab[mention[2]][mention[0]][mention[1]] = 1
                    info["mention"][sentence_id].add(mention)

            assert len(mentions) == len(instance["gold"]["trigger"]) + len(instance["gold"]["argument"]) + len(instance["gold"]["network"])
            labs.append(lab)
            info["sent_idx"].append(instance["sent_id"])

    input_ids = torch.LongTensor(input_ids)
    ent_labs = torch.LongTensor(ent_labs)
    offset = torch.LongTensor(offsets)
    labs = torch.LongTensor(labs)
    seq_lens = torch.LongTensor(seq_lens)

    return info, input_ids, offset, seq_lens, ent_labs, labs
