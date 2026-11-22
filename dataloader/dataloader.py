import json
import os
import re
import torch
import math
import random
import numpy as np
from collections import Counter
from tqdm import tqdm


class SpanDataloader:
    def __init__(self,
                 train_path,
                 devel_path,
                 test_path,
                 task_name,
                 tokenizer=None,
                 logger=None, ):
        self.train_path = train_path
        self.devel_path = devel_path
        self.test_path = test_path
        self.tokenizer = tokenizer
        self.logger = logger
        self.task_name = task_name

    def load_data(self, file_path, dataset_name):
        with open(file_path, encoding='utf-8') as f:
            lines = f.readlines()
            lines = [json.loads(ele) for ele in tqdm(lines, desc=f"loading {dataset_name} instances")]
        return lines

    def build_map(self):
        if not os.path.exists(f"experiments/{self.task_name}/schema.json"):
            data = self.load_data(file_path=self.train_path, dataset_name="train")
            train_data = data
            data += self.load_data(file_path=self.devel_path, dataset_name="devel")
            data += self.load_data(file_path=self.test_path, dataset_name="test")
            deps, tags, trigger, entity, rels, structs = [], [], [], [], [], {}
            for instance in train_data:
                for item in instance["relation"]:
                    if item["role"] not in structs:
                        structs[item["role"]] = []
                    t_type = item["trigger"]["type"]
                    a_type = item["argument"]["type"]
                    if (t_type, a_type) not in structs[item["role"]]:
                        structs[item["role"]].append((t_type, a_type))
            for instance in data:
                deps.extend([item[1] for item in instance["token"]["mask_dep"]])
                tags.extend(instance["token"]["mask_tag"])
                tris = instance["trigger"]
                ents = instance["entity"]
                rels.extend(instance["rel_lab"])
                for tri in tris.values():
                    # if tri["type"] in all_tri_type:
                    trigger.append(tri["type"])
                for ent in ents.values():
                    entity.append(ent["type"])

            dep_tuple = ["O"] + [item[0] for item in
                                 sorted(dict(Counter(deps)).items(), key=lambda x: x[1], reverse=True)] + ["self-loop"]
            tag_tuple = ["O"] + [item[0] for item in
                                 sorted(dict(Counter(tags)).items(), key=lambda x: x[1], reverse=True)]
            tri_tuple = ["O"] + [item[0] for item in
                                 sorted(dict(Counter(trigger)).items(), key=lambda x: x[1], reverse=True)]

            rel_tuple = [item[0] for item in
                         sorted(dict(Counter(rels)).items(), key=lambda x: x[1], reverse=True)]
            # kkk = sorted(dict(Counter(rels)).items(), key=lambda x: x[1], reverse=True)

            ent_tuple = ["O"]
            for item in [item[0] for item in sorted(dict(Counter(entity)).items(), key=lambda x: x[1], reverse=True)]:
                ent_tuple.append(f"B-{item}")
                ent_tuple.append(f"I-{item}")
            lab_tuple = ["O"]
            for lab in tri_tuple[1:]:
                lab_tuple.append(f"B-{lab}")
                lab_tuple.append(f"I-{lab}")

            dep2id = {key: i for i, key in enumerate(dep_tuple)}
            tag2id = {key: i for i, key in enumerate(tag_tuple)}
            tri2id = {key: i for i, key in enumerate(tri_tuple)}
            ent2id = {key: i for i, key in enumerate(ent_tuple)}
            lab2id = {key: i for i, key in enumerate(lab_tuple)}
            rel2id = {key: i for i, key in enumerate(rel_tuple)}

            schema = {"tri": tri2id, "ent": ent2id, "dep": dep2id, "tag": tag2id,
                      "lab": lab2id, "rel": rel2id, "struct": structs}
            with open(f"experiments/{self.task_name}/schema.json", encoding="utf-8", mode="w") as f:
                json.dump(schema, f, ensure_ascii=False, indent=4)
        else:
            schema = json.load(open(f"experiments/{self.task_name}/schema.json", encoding="utf-8", mode="r"))
        return schema

    def get_tensor(self, data_path, dataset_name, schema):
        all_data = self.load_data(file_path=data_path, dataset_name="train")
        ent2id, lab2id, tag2id, dep2id, rel2id = schema["ent"], schema["lab"], schema["tag"], schema["dep"], schema[
            "rel"]
        pp_data, trigger_count, entity_count, total_roles, contain_roles, pos, neg = [], 0, 0, 0, 0, 0, 0
        t, r, e = set(), set(), set()
        for i in tqdm(range(len(all_data)), desc=f"generating {dataset_name} instances"):
            instance = all_data[i]
            sent_id = instance["id"]
            doc_id = sent_id.split(":s")[0]
            tokens = [item[3] for item in instance["token"]["mask_token"]]
            char_map = [[item[0], item[1]] for item in instance["token"]["mask_token"]]
            tris = instance["trigger"]
            ents = instance["entity"]
            relations = instance["relation"]
            # tag = [tag2id[item] for item in instance["token"]["mask_tag"]]
            tag = instance["token"]["mask_tag"]
            dep = [[item[0], dep2id[item[1]], item[-1]] for item in instance["token"]["mask_dep"]]
            rev_dep = [[item[-1], item[1], item[0]] for item in dep] + [[0, dep2id["self-loop"], 0]]
            dep += rev_dep

            entities_str = []
            ent_lab = ["O"] * len(tokens)
            trigger_count += len(tris)
            entity_count += len(ents)
            for ent in ents.values():
                text = ent["text"]
                type_ = ent["type"]
                char_start = ent["doc_start"]
                char_end = ent["doc_end"]
                mask_start = ent["mask_start"]
                mask_end = ent["mask_end"]
                ent_str = f"{doc_id}&&{char_start}-{char_end}:{text}>>{type_}"
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
            triggers_str = []
            tlab = [lab2id[item] for item in instance["tri_lab"]]
            rlab = [rel2id[item] for item in instance["rel_lab"]]
            lab = 0
            if instance["txt_lab"] != "O":
                lab = 1
            for tri in tris.values():
                trigger_start, trigger_end, trigger_text, type_ = tri["mask_start"], tri["mask_end"], tri["text"], tri[
                    "type"]
                if self.task_name in ["GE13", "GE11"]:
                    char_start, char_end = char_map[trigger_start][0], char_map[trigger_start][-1]
                else:
                    char_start, char_end = char_map[trigger_start][0], char_map[trigger_end - 1][-1]
                trigger_text = org_txt[char_start:char_end]
                trigger_str = f"{doc_id}&&{char_start + sent_start}-{char_end + sent_start}:{trigger_text}>>{type_}"
                # if trigger_str not in triggers_str and type_ in all_tri_type:
                if trigger_str not in triggers_str:
                    triggers_str.append(trigger_str)
                    t.add(trigger_str)
            rel_str = set()
            for relation in relations:
                trigger_start, trigger_end, trigger_type = relation["trigger"]["mask_start"], relation["trigger"][
                    "mask_end"], relation["trigger"]["type"]
                if self.task_name in ["GE11", "GE13"]:
                    t_char_start, t_char_end = char_map[trigger_start][0], char_map[trigger_start][-1]
                else:
                    t_char_start, t_char_end = char_map[trigger_start][0], char_map[trigger_end - 1][-1]
                trigger_text = org_txt[t_char_start:t_char_end]

                argument_start, argument_end, argument_type = relation["argument"]["mask_start"], relation["argument"][
                    "mask_end"], relation["argument"]["type"]
                if self.task_name in ["GE11", "GE13"]:
                    a_char_start, a_char_end = char_map[argument_start][0], char_map[argument_start][-1]
                else:
                    a_char_start, a_char_end = char_map[argument_start][0], char_map[argument_end - 1][-1]

                argument_text = org_txt[a_char_start:a_char_end]
                role = relation["role"]

                temp = f"{doc_id}&&{t_char_start + sent_start}-{t_char_end + sent_start}:{trigger_text}>>{trigger_type}->{role}<-{argument_type}<<{a_char_start + sent_start}-{a_char_end + sent_start}:{argument_text}"
                rel_str.add(temp)
                r.add(temp)

            token_sent = [self.tokenizer.cls_token]
            bert_span = []
            offsets = []
            bert_start, bert_end = 1, 0
            for token_index, token in enumerate(tokens + instance["text_b"].split(" ")):
                bert_tk = self.tokenizer.tokenize(token)
                bert_end = bert_start + len(bert_tk)
                token_sent.extend(bert_tk)
                bert_span.append((bert_start, bert_end))
                offsets.append(bert_start)
                bert_start = bert_end

            token_sent.append(self.tokenizer.sep_token)

            bert_tk_id = self.tokenizer.encode(' '.join(tokens + instance["text_b"].split(" ")))

            assert len(token_sent) == len(bert_tk_id)
            assert len(tokens + instance["text_b"].split(" ")) == len(offsets)
            len_t = instance["length"]
            if lab == 0:
                neg += 1
            else:
                pos += 1
            pp_data.append(
                {
                    "sent_id": sent_id,
                    "doc_id": doc_id,
                    "text": tokens,
                    "sent_start": sent_start,
                    "org_txt": org_txt,
                    "bert":
                        {"tokens": tokens, "bert-tokens": token_sent, "input_ids": bert_tk_id,
                         "offsets": offsets[:len_t]},
                    "trigger": triggers_str,
                    "entity": entities_str,
                    "relation": rel_str,
                    "postag": instance["token"]["mask_tag"],
                    "char_map": char_map,
                    "tri": tlab,
                    "ent": ent_lab,
                    "tag": tag,
                    "dep": dep,
                    "rel": rlab,
                    "lab": lab,
                    "len": len_t,
                    "pos": instance["position"]
                }
            )
        info = [f"trigger count: {len(t)}", f"entity count: {len(e)}",
                f"pos/neg: {pos}/{neg}", f"relation count: {len(r)}"]
        return pp_data, info

    def get_batch_data(self, data, mode, batch_size):
        self.num_batch = int(math.ceil(len(data) / batch_size))
        sorted_data = sorted(data, key=lambda x: len(x['bert']['input_ids']), reverse=True)
        batch_data = list()
        for i in range(self.num_batch):
            batch_data.append(self.pad_data(sorted_data[i * batch_size: (i + 1) * batch_size]))
        return batch_data

    def pad_data(self, data):
        input_ids, idx, labs = [], [], []
        offsets, tri_labs, rel_labs = [], [], []
        ent_labs, graphs = [], []
        info = {"seq_len": [], "sent_idx": [], "doc_idx": [], "org_txt": [], "char_map": [], "tg": [], "rg": [],
                "sentText": [], "sent_start": [], "pos": [], "ent": [], "tag": []}
        max_bert_length = max([len(item['bert']['input_ids']) for item in data])
        max_token_length = max([len(item['bert']['offsets']) for item in data])
        for line in data:
            bert_info = line['bert']
            input_id = bert_info['input_ids']
            offset = bert_info['offsets']
            doc_id = line["doc_id"]
            sent_id = line["sent_id"]
            deps = line["dep"]

            info["org_txt"].append(line["org_txt"])
            info["seq_len"].append(len(offset))
            info["sentText"].append(line["text"])
            info["sent_start"].append(line["sent_start"])

            info["char_map"].append(line["char_map"])
            info["doc_idx"].append(doc_id)
            info["sent_idx"].append(sent_id)
            info["tg"].append(line["trigger"])
            info["rg"].append(line["relation"])
            info["pos"].append(line["pos"])
            info["ent"].append(line["entity"])
            info["tag"].append(line["tag"])

            tri_lab = line["tri"]
            ent_lab = line["ent"]
            rel_lab = line["rel"]
            lab = line["lab"]
            if len(input_id) <= max_bert_length:  # Bert补全
                pad_token = [0] * (max_bert_length - len(input_id))
                input_ids.append(input_id + pad_token)
            if len(offset) <= max_token_length:
                pad_token = [0] * (max_token_length - len(offset))
                offsets.append(offset + pad_token)
                tri_labs.append(tri_lab + pad_token)
                rel_labs.append(rel_lab + pad_token)
                ent_labs.append(ent_lab + pad_token)
            labs.append(lab)
            graph = np.zeros(shape=[max_token_length, max_token_length], dtype=np.int64)
            for dep_item in deps[:-1]:
                graph[dep_item[0], dep_item[-1]] = dep_item[1]
                graph[dep_item[-1], dep_item[0]] = dep_item[1]

            for i in range(max_token_length):
                graph[i, i] = deps[-1][1]

            graphs.append(graph)

        input_ids = torch.LongTensor(input_ids)
        ent_labs = torch.LongTensor(ent_labs)
        offset = torch.LongTensor(offsets)
        tri_labs = torch.LongTensor(tri_labs)
        rel_labs = torch.LongTensor(rel_labs)
        graphs = torch.LongTensor(graphs)
        labs = torch.LongTensor(labs)
        seq_lens = torch.LongTensor(info["seq_len"])
        return [input_ids, offset, ent_labs, graphs, tri_labs, rel_labs, labs, info, seq_lens]

    def iter_batch(self, batch_data, shuffle=False):
        if shuffle:
            random.shuffle(batch_data)
        len_data = len(batch_data)
        for idx in range(len_data):
            yield batch_data[idx]


import os
import json
import torch
import random
from functools import partial
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, raw_data, label_dict, method):
        dataset = list()
        for data in raw_data:
            tokens = data['text'].lower().split(' ')
            label_id = label_dict[data['label']]
            dataset.append((tokens, label_id))
        self._dataset = dataset
        self._method = method
        self.sep_token = ['[SEP]']
        self._label_list = list(label_dict.keys())
        self._num_classes = len(label_dict)

    def __getitem__(self, index):
        tokens, label_id = self._dataset[index]
        if self._method not in ['ce', 'scl']:
            rand_idx = [i for i in range(self._num_classes)]
            random.shuffle(rand_idx)
            label_list = [self._label_list[i] for i in rand_idx]
            tokens = label_list + self.sep_token + tokens
            label_id = rand_idx[label_id]
        return tokens, label_id

    def __len__(self):
        return len(self._dataset)



def my_collate(batch, tokenizer):
    tokens, label_ids = map(list, zip(*batch))
    text_ids = tokenizer(tokens,
                         padding=True,
                         truncation=True,
                         max_length=256,
                         is_split_into_words=True,
                         add_special_tokens=True,
                         return_tensors='pt')
    return text_ids, torch.tensor(label_ids)