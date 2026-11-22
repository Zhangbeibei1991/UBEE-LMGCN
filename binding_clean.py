import os
import pickle
import json
import copy
from transformers import AutoTokenizer, AutoModel

tokenizer = AutoTokenizer.from_pretrained("./bert/scibert_scivocab_uncased")

if not os.path.exists("binding.pt"):
    train_pt = pickle.load(open("sources/GE13/train.pt", mode="rb"))
    devel_pt = pickle.load(open("sources/GE13/devel.pt", mode="rb"))
    test_pt = pickle.load(open("sources/GE13/test.pt", mode="rb"))

    binding_train = []
    binding_devel = []

    # 统计触发词和要素之间的距离，如果有两个Theme也要统计两个theme之间的距离发那我
    binding1_sta, binding2_sta = set(), set()
    binding2_sp = set()
    for sample in train_pt + devel_pt:
        triggers = sample["trigger"]
        tokens = [item[-1] for item in sample["token"]["mask_token"]]
        flag = False
        for t_id, trigger in triggers.items():
            if trigger["type"] == "Binding":
                flag = True
                break
        if flag:
            idx = sample["id"]
            events = sample["event"]
            entities = sample["entity"]
            triggers = {key: value for key, value in sample["trigger"].items() if value["type"] == "Binding"}
            evt_map = {}
            for e_id, evt in events.items():
                if evt["type"] == "Binding":
                    id_content = evt["id_content"]
                    tri_token_start, tri_char_start = id_content["mask_start"], id_content["doc_start"]
                    tri_token_end, tri_char_end = id_content["mask_start"], id_content["doc_start"]
                    tri_txt = id_content["text"]
                    arg_roles = evt["arg_roles"]
                    arg_contents = evt["arg_content"]
                    arg_list = []
                    for role, arg_content in zip(arg_roles, arg_contents):
                        arg_token_start, arg_char_start = arg_content["mask_start"], arg_content["doc_start"]
                        arg_token_end, arg_char_end = arg_content["mask_end"], arg_content["doc_end"]
                        arg_txt = arg_content["text"]
                        if role[-1].isdigit():
                            role = role[:-1]
                        arg_list.append((arg_token_start, arg_token_end, arg_char_start, arg_char_end, arg_txt, role))
                        if len(arg_roles) == 1:
                            binding1_sta.add(tri_char_start - arg_char_start)

                        else:
                            binding2_sta.add(tri_char_start - arg_char_start)
                    if len(arg_roles) == 2:
                        binding2_sp.add(arg_list[0][2] - arg_list[1][2])
    binding1_max, binding1_min = max(binding1_sta), min(binding1_sta)
    binding2_max, binding2_min = max(binding2_sta), min(binding2_sta)
    binding2s_max, binding2s_min = max(binding2_sp), min(binding2_sp)
    pos, neg, max_len = 0, 0, 0
    all_event_num = 0
    for sample in train_pt:
        triggers = sample["trigger"]
        tokens = [item[-1] for item in sample["token"]["mask_token"]]
        flag = False
        for t_id, trigger in triggers.items():
            if trigger["type"] == "Binding":
                flag = True
                break
        if flag:
            idx = sample["id"]
            events = sample["event"]
            entities = sample["entity"]
            triggers = {key: value for key, value in sample["trigger"].items() if value["type"] == "Binding"}
            evt_map = {}
            binding_themes1, binding_themes2 = [], []
            for e_id, evt in events.items():
                if evt["type"] == "Binding":
                    all_event_num += 1
                    id_content = evt["id_content"]
                    tri_token_start, tri_char_start = id_content["mask_start"], id_content["doc_start"]
                    tri_token_end, tri_char_end = id_content["mask_start"], id_content["doc_start"]
                    tri_txt = id_content["text"]
                    arg_roles = evt["arg_roles"]
                    arg_contents = evt["arg_content"]
                    arg_list = []
                    for role, arg_content in zip(arg_roles, arg_contents):
                        arg_token_start, arg_char_start = arg_content["mask_start"], arg_content["doc_start"]
                        arg_token_end, arg_char_end = arg_content["mask_end"], arg_content["doc_end"]
                        arg_txt = arg_content["text"]
                        if role[-1].isdigit():
                            role = role[:-1]
                            binding_themes2.append(arg_content)
                        else:
                            binding_themes1.append(arg_content)
                        arg_list.append((arg_token_start, arg_token_end, arg_char_start, arg_char_end, arg_txt, role))
                    arg_list = sorted(arg_list, key=lambda x: x[0], reverse=False)
                    evt_str = f"{idx}&&{tri_char_start}-{tri_char_end}:{tri_txt}>>Binding$$"
                    for item in arg_list:
                        evt_str += f"{item[2]}-{item[3]}:{item[4]}>>{item[5]}==>"
                    evt_str = evt_str.strip("==>")
                    evt_list = [(tri_token_start, tri_token_end, tri_char_start, tri_char_end, tri_txt,
                                 "Binding")] + arg_list
                    evt_map[evt_str] = evt_list
            # Binding1 event
            for tri_id, trigger in triggers.items():
                for entity in binding_themes1:
                    tri_token_start, tri_char_start = trigger["mask_start"], trigger["doc_start"]
                    tri_token_end, tri_char_end = trigger["mask_start"], trigger["doc_start"]
                    tri_txt = trigger["text"]

                    arg_token_start, arg_char_start = entity["mask_start"], entity["doc_start"]
                    arg_token_end, arg_char_end = entity["mask_end"], entity["doc_end"]
                    arg_txt = entity["text"]

                    evt_list = [(tri_token_start, tri_token_end, tri_char_start, tri_char_end, tri_txt, "Binding"),
                                (arg_token_start, arg_token_end, arg_char_start, arg_char_end, arg_txt, "Theme")]

                    evt_str = f"{idx}&&{tri_char_start}-{tri_char_end}:{tri_txt}>>Binding$${arg_char_start}-{arg_char_end}:{arg_txt}>>Theme"

                    lab = "negative"
                    if evt_str in evt_map:
                        lab = "positive"
                        pos += 1
                    else:
                        neg += 1

                    event_description = f"Is the #Binding# type flat event consisting a trigger ${tri_txt}$ at position {tri_token_start}-{tri_token_end} , " \
                                        f"a @Theme@ $Protein$ event argument ${arg_txt}$ at position {arg_token_start}-{arg_token_end} correct ?"

                    inst_tokens = tokens + event_description.split()
                    bert_input_ids = tokenizer.encode(" ".join(inst_tokens))
                    max_len = max(max_len, len(bert_input_ids))
                    binding_train.append([sample, inst_tokens, bert_input_ids, evt_list, lab])
            # Binding2 event
            for tri_id, trigger in triggers.items():
                for i, entity1 in enumerate(binding_themes1):
                    for j, entity2 in enumerate(binding_themes2):
                        if i > j:
                            continue
                        tri_token_start, tri_char_start = trigger["mask_start"], trigger["doc_start"]
                        tri_token_end, tri_char_end = trigger["mask_start"], trigger["doc_start"]
                        tri_txt = trigger["text"]

                        arg_token_start1, arg_char_start1 = entity1["mask_start"], entity1["doc_start"]
                        arg_token_end1, arg_char_end1 = entity1["mask_end"], entity1["doc_end"]
                        arg_txt1 = entity1["text"]

                        arg_token_start2, arg_char_start2 = entity2["mask_start"], entity2["doc_start"]
                        arg_token_end2, arg_char_end2 = entity2["mask_end"], entity2["doc_end"]
                        arg_txt2 = entity2["text"]

                        arg_list = [
                            (arg_token_start1, arg_token_end1, arg_char_start1, arg_char_end1, arg_txt1, "Theme"),
                            (arg_token_start2, arg_token_end2, arg_char_start2, arg_char_end2, arg_txt2, "Theme")]
                        arg_list = sorted(arg_list)

                        evt_list = [(tri_token_start, tri_token_end, tri_char_start, tri_char_end, tri_txt,
                                     "Binding")] + arg_list

                        evt_str = f"{idx}&&{tri_char_start}-{tri_char_end}:{tri_txt}>>Binding$$"
                        for item in arg_list:
                            evt_str += f"{item[2]}-{item[3]}:{item[4]}>>{item[5]}==>"
                        evt_str = evt_str.strip("==>")

                        lab = "negative"
                        if evt_str in evt_map:
                            lab = "positive"
                            pos += 1
                        else:
                            neg += 1

                        event_description = f"Is the #Binding# type flat event consisting a trigger ${tri_txt}$ at position {tri_token_start}-{tri_char_start} , " \
                                            f"a @Theme@ $Protein$ event argument ${arg_list[0][4]}$ at position {arg_list[0][2]}-{arg_list[0][3]} and " \
                                            f"a @Theme@ $Protein$ event argument ${arg_list[1][4]}$ at position {arg_list[1][2]}-{arg_list[1][3]} correct ?"

                        inst_tokens = tokens + event_description.split()
                        bert_input_ids = tokenizer.encode(" ".join(inst_tokens))
                        max_len = max(max_len, len(bert_input_ids))
                        binding_train.append([sample, inst_tokens, bert_input_ids, evt_list, lab])
    print(f"in train, the proportion of pos/neg instance is: {pos}/{neg}| max len: {max_len} >> {all_event_num}")
    pos, neg, max_len = 0, 0, 0
    all_event_num = 0
    for sample in devel_pt:
        triggers = sample["trigger"]
        tokens = [item[-1] for item in sample["token"]["mask_token"]]
        flag = False
        for t_id, trigger in triggers.items():
            if trigger["type"] == "Binding":
                flag = True
                break
        if flag:
            idx = sample["id"]
            events = sample["event"]
            entities = sample["entity"]
            triggers = {key: value for key, value in sample["trigger"].items() if value["type"] == "Binding"}
            evt_map = {}
            binding_themes1, binding_themes2 = [], []
            for e_id, evt in events.items():
                if evt["type"] == "Binding":
                    all_event_num += 1
                    id_content = evt["id_content"]
                    tri_token_start, tri_char_start = id_content["mask_start"], id_content["doc_start"]
                    tri_token_end, tri_char_end = id_content["mask_end"], id_content["doc_end"]
                    tri_txt = id_content["text"]
                    arg_roles = evt["arg_roles"]
                    arg_contents = evt["arg_content"]
                    arg_list = []
                    for role, arg_content in zip(arg_roles, arg_contents):
                        arg_token_start, arg_char_start = arg_content["mask_start"], arg_content["doc_start"]
                        arg_token_end, arg_char_end = arg_content["mask_end"], arg_content["doc_end"]
                        arg_txt = arg_content["text"]
                        if role[-1].isdigit():
                            role = role[:-1]
                            binding_themes2.append(arg_content)
                        else:
                            binding_themes1.append(arg_content)
                        arg_list.append((arg_token_start, arg_token_end, arg_char_start, arg_char_end, arg_txt, role))

                    arg_list = sorted(arg_list, key=lambda x: x[0], reverse=False)
                    evt_str = f"{idx}&&{tri_char_start}-{tri_char_end}:{tri_txt}>>Binding$$"
                    for item in arg_list:
                        evt_str += f"{item[2]}-{item[3]}:{item[4]}>>{item[5]}==>"
                    evt_str = evt_str.strip("==>")
                    evt_list = [(tri_token_start, tri_token_end, tri_char_start, tri_char_end, tri_txt,
                                 "Binding")] + arg_list
                    evt_map[evt_str] = evt_list
            # Binding1 event
            for tri_id, trigger in triggers.items():
                for entity in binding_themes1:
                    tri_token_start, tri_char_start = trigger["mask_start"], trigger["doc_start"]
                    tri_token_end, tri_char_end = trigger["mask_end"], trigger["doc_end"]
                    tri_txt = trigger["text"]

                    arg_token_start, arg_char_start = entity["mask_start"], entity["doc_start"]
                    arg_token_end, arg_char_end = entity["mask_end"], entity["doc_end"]
                    arg_txt = entity["text"]

                    evt_list = [(tri_token_start, tri_token_end, tri_char_start, tri_char_end, tri_txt, "Binding"),
                                (arg_token_start, arg_token_end, arg_char_start, arg_char_end, arg_txt, "Theme")]

                    evt_str = f"{idx}&&{tri_char_start}-{tri_char_end}:{tri_txt}>>Binding$${arg_char_start}-{arg_char_end}:{arg_txt}>>Theme"

                    lab = "negative"
                    if evt_str in evt_map:
                        lab = "positive"
                        pos += 1
                    else:
                        neg += 1

                    event_description = f"Is the #Binding# type flat event consisting a trigger ${tri_txt}$ at position {tri_token_start}-{tri_token_end} , " \
                                        f"a @Theme@ $Protein$ event argument ${arg_txt}$ at position {arg_token_start}-{arg_token_end} correct ?"

                    inst_tokens = tokens + event_description.split()
                    bert_input_ids = tokenizer.encode(" ".join(inst_tokens))
                    max_len = max(max_len, len(bert_input_ids))
                    binding_devel.append([sample, inst_tokens, bert_input_ids, evt_list, lab])
            # Binding2 event
            for tri_id, trigger in triggers.items():
                for i, entity1 in enumerate(binding_themes1):
                    for j, entity2 in enumerate(binding_themes2):
                        if i > j:
                            continue
                        tri_token_start, tri_char_start = trigger["mask_start"], trigger["doc_start"]
                        tri_token_end, tri_char_end = trigger["mask_end"], trigger["doc_end"]
                        tri_txt = trigger["text"]

                        arg_token_start1, arg_char_start1 = entity1["mask_start"], entity1["doc_start"]
                        arg_token_end1, arg_char_end1 = entity1["mask_end"], entity1["doc_end"]
                        arg_txt1 = entity1["text"]

                        arg_token_start2, arg_char_start2 = entity2["mask_start"], entity2["doc_start"]
                        arg_token_end2, arg_char_end2 = entity2["mask_end"], entity2["doc_end"]
                        arg_txt2 = entity2["text"]

                        arg_list = [
                            (arg_token_start1, arg_token_end1, arg_char_start1, arg_char_end1, arg_txt1, "Theme"),
                            (arg_token_start2, arg_token_end2, arg_char_start2, arg_char_end2, arg_txt2, "Theme")]
                        arg_list = sorted(arg_list)

                        evt_list = [(tri_token_start, tri_token_end, tri_char_start, tri_char_end, tri_txt,
                                     "Binding")] + arg_list

                        evt_str = f"{idx}&&{tri_char_start}-{tri_char_end}:{tri_txt}>>Binding$$"
                        for item in arg_list:
                            evt_str += f"{item[2]}-{item[3]}:{item[4]}>>{item[5]}==>"
                        evt_str = evt_str.strip("==>")

                        lab = "negative"
                        if evt_str in evt_map:
                            lab = "positive"
                            pos += 1
                        else:
                            neg += 1

                        event_description = f"Is the #Binding# type flat event consisting a trigger ${tri_txt}$ at position {tri_token_start}-{tri_char_start} , " \
                                            f"a @Theme@ $Protein$ event argument ${arg_list[0][4]}$ at position {arg_list[0][2]}-{arg_list[0][3]} and " \
                                            f"a @Theme@ $Protein$ event argument ${arg_list[1][4]}$ at position {arg_list[1][2]}-{arg_list[1][3]} correct ?"

                        inst_tokens = tokens + event_description.split()
                        bert_input_ids = tokenizer.encode(" ".join(inst_tokens))
                        max_len = max(max_len, len(bert_input_ids))
                        binding_devel.append([sample, inst_tokens, bert_input_ids, evt_list, lab])
    print(f"in devel, the proportion of pos/neg instance is: {pos}/{neg}| max len: {max_len} >> {all_event_num}")
    with open("binding.pt", mode="wb") as f:
        pickle.dump([binding_train, binding_devel], f)
else:
    binding_train, binding_devel = pickle.load(open("binding.pt", mode="rb"))
# 构造句子分类样本
lab2id = {"positive": 0, "negative": 1}

import torch
import torch.nn as nn
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):
    def __init__(self, raw_data):
        dataset = []
        for data in raw_data:
            dataset.append((data[2], lab2id[data[4]]))
        self._dataset = dataset

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


def my_collate(batch):
    input_ids_, label_ids = map(list, zip(*batch))
    max_len = max([len(input_id) for input_id in input_ids_])
    input_ids = []
    for input_id in input_ids_:
        pad_token = [0] * (max_len - len(input_id))
        input_ids.append(input_id + pad_token)
    return torch.LongTensor(input_ids), torch.tensor(label_ids)


trainset = MyDataset(binding_train)
develset = MyDataset(binding_devel)

encoder = AutoModel.from_pretrained("./bert/scibert_scivocab_uncased")


class BindingClean(nn.Module):
    def __init__(self):
        super(BindingClean, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Linear(768, 2)

    def forward(self, input_ids):
        attention_mask = (input_ids > 0).long()
        encode = self.encoder(input_ids, attention_mask).last_hidden_state
        cls = encode[:, 0, :]
        prob = self.classifier(cls)
        return prob


epoch = 300
lr = 1e-5
train_batch_size = 8
test_batch_size = 8
workers = 0

model = BindingClean()

_params = filter(lambda p: p.requires_grad, model.parameters())
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(_params, lr=lr)

train_dataloader = DataLoader(trainset, train_batch_size, shuffle=True, num_workers=workers, collate_fn=my_collate,
                              pin_memory=True)
devel_dataloader = DataLoader(develset, test_batch_size, shuffle=False, num_workers=workers, collate_fn=my_collate,
                              pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(epoch):
    model.train()
    best_f, best_epoch = 0.0, 0
    for inputs, targets in tqdm(train_dataloader, disable=False, ascii=' >='):
        inputs = inputs.to(device)
        targets = targets.to(device)
        prob = model(inputs)
        loss = criterion(prob, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    preds, golds = set(), set()
    count = 0
    for inputs, targets in tqdm(devel_dataloader, disable=False, ascii=' >='):
        inputs = inputs.to(device)
        targets = targets.to(device)
        prob = model(inputs)
        prob_idx = torch.argmax(prob, dim=-1).tolist()
        for idx in range(len(prob_idx)):
            if prob_idx[idx] == 0:
                preds.add((count, "positive"))
            if targets[idx] == 0:
                golds.add((count, "positive"))
            count += 1
    corr_num = len(preds & golds)
    pred_num = len(preds)
    gold_num = len(golds)
    P = corr_num / (pred_num + 1e-10)
    R = corr_num / (gold_num + 1e-10)
    F = 2 * P * R / (P + R + 1e-10)

    print(f"P: {P}")
    print(f"R: {R}")
    print(f"F: {F}")

    if best_f < F:
        best_f = F
        best_epoch = epoch
        torch.save(model.state_dict(), "binding_model.pt")


print()
