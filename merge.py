import os
import shutil
import json
import re
import subprocess

Theme_events = ["Gene_expression", "Transcription", "Localization", "Protein_catabolism"]
Theme_Cause_events = ["Regulation", "Negative_regulation", "Positive_regulation", "Phosphorylation"]
Other_events = ["Protein_modification", "Ubiquitination", "Deacetylation", "Acetylation"]
MThemes_events = ["Binding"]
Nested_events = ["Regulation", "Negative_regulation", "Positive_regulation"]

structure = json.load(open("./sources/GE13/structure.json", encoding="utf-8", mode="r"))


def merge(aux, source_path, save_path, mode="devel", model_name="GE13", logger=None):
    trigger_p, trigger_g, relation_p, relation_g, network_p, network_g, entity_g = aux
    whole_names = sorted([item[:-4] for item in os.listdir(os.path.join(source_path, mode)) if
                          item.endswith(".txt")])
    os.makedirs(os.path.join(save_path, mode, "a1"), exist_ok=True)
    os.makedirs(os.path.join(save_path, mode, "a2"), exist_ok=True)
    os.makedirs(os.path.join(save_path, mode, "txt"), exist_ok=True)
    for i, name in enumerate(whole_names):
        max_ent_num = 0
        shutil.copy(os.path.join(source_path, mode, name + ".txt"), os.path.join(save_path, mode, "txt", name + ".txt"))
        name = name.replace("NF-+%A6B", "NF-κB")
        if mode == "devel":
            shutil.copy(os.path.join(source_path, mode, name + ".a2"),
                        os.path.join(save_path, mode, "a1", name + ".a2"))
        ent_map = {}
        mix = {}
        with open(os.path.join(save_path, mode, "a1", name + ".a1"), encoding="utf-8", mode="w") as f:
            if name in entity_g:
                entities = entity_g[name]
                ent_std = []
                for entity in entities:
                    ent_id = entity.split("&&")[0].split(".")[-1]
                    span = entity.split("&&")[1].split(":")[0].replace("-", " ")
                    txt = entity.split("&&")[1].split(":")[1].split(">>")[0]
                    type_ = entity.split(">>")[-1]
                    ent_std.append((ent_id, type_, span, txt))
                    ent_map[ent_id] = (ent_id, type_, span, txt)
                ent_std = sorted(ent_std, key=lambda x: int(x[2].split()[0]))
                for item in ent_std:
                    mix[item[0]] = item
                    f.write(f"{item[0]}\t{item[1]} {item[2]}\t{item[3]}\n")
                try:
                    max_ent_num = max([int(item_[1:]) for item_ in ent_map.keys()]) + 1
                except:
                    assert ("error!")
        with open(os.path.join(save_path, mode, "a2", name + ".a2"), encoding="utf-8", mode="w") as f:
            if name in trigger_p:
                triggers = trigger_p[name]
                tri_std = []
                tri_map = {}
                for trigger in triggers:
                    span = trigger.split("&&")[1].split(":")[0].replace("-", " ")
                    span_txt = trigger.split(":")[1].split(">>")[0]
                    tri_start, tri_end = int(span.split(" ")[0]), int(span.split(" ")[1])
                    if tri_end - tri_start > 100:
                        span_txt = span_txt.split()[0]
                        tri_end = tri_start + len(txt)
                        span = f"{tri_start} {tri_end}"
                    type_ = trigger.split(">>")[-1]
                    if [type_, span, span_txt] not in tri_std:
                        tri_std.append([type_, span, span_txt])
                tri_std = sorted(tri_std, key=lambda x: int(x[1].split()[0]))
                for i in range(len(tri_std)):
                    tri_std[i].insert(0, f"T{max_ent_num}")
                    assert str(tri_std[i][0][0]) in ['T', 'R', "E", 'M']
                    tri_map[f"T{max_ent_num}"] = tri_std[i]
                    max_ent_num += 1
                for item in tri_std:
                    mix[item[0]] = item
                    f.write(f"{item[0]}\t{item[1]} {item[2]}\t{item[3]}\n")
                for item in ent_std:
                    try:
                        assert item[0] not in tri_map
                        assert str(item[0][0]) in ['T', 'R', "E", 'M']
                    except:
                        print()
            else:
                f.write("")
            # 融合事件
            # STEP-1: 融合简单事件
            extracted_relation = None
            extracted_network = None
            network_map = {}
            if name in relation_p:
                extracted_relation = relation_p[name]
            if name in network_p:
                extracted_network = network_p[name]
                # 将要素链接转化成id索引的模式
                network_map = {}
                for network in extracted_network:
                    head_span = network.split("&&")[1].split(":")[0].replace("-", " ")
                    head_type = network.split(">>")[1].split("->")[0]
                    head_text = network.split(":")[1].split(">>")[0]
                    head_role = network.split("->")[1].split("$")[0]
                    tail_role = network.split("->")[1].split("$")[1].split("|")[0]
                    tail_type = network.split("<-")[1].split("<<")[0]
                    tail_text = network.split("<<")[1].split(":")[0]
                    temp = network.split(":")[-1].split("-")
                    tail_span = f"{temp[1]} {temp[0]}"
                    head_id, tail_id = None, None
                    for m_id, m_tuple in mix.items():
                        if m_tuple[1] == head_type and m_tuple[2] == head_span:
                            head_id = m_id
                            break
                    if head_id is None:
                        continue
                    for m_id, m_tuple in mix.items():
                        if m_tuple[1] == tail_type and m_tuple[2] == tail_span:
                            tail_id = m_id
                            break
                    if tail_id is None:
                        continue
                    temp = (head_id, head_type, head_span, head_text)
                    if temp not in network_map:
                        network_map[temp] = set()
                    network_map[temp].add((head_role, tail_role, tail_id, tail_type, tail_span, tail_text))
            if extracted_relation is not None:
                simple_relations = []
                for relation in extracted_relation:
                    tri_span = relation.split("&&")[1].split(":")[0].replace("-", " ")
                    tri_type = relation.split(">>")[1].split("->")[0]
                    role = relation.split(">>")[1].split("->")[1].split("<-")[0].split("$")[0]
                    arg_type = relation.split("<-")[1].split("<<")[0]
                    arg_start, arg_end = relation.split(":")[-1].split("-")
                    arg_span = f"{arg_start} {arg_end}"
                    if tri_type in Theme_events + ["Phosphorylation"]:
                        tri_id = None
                        for t_id, t_tuple in tri_map.items():
                            if t_tuple[1] == tri_type and t_tuple[2] == tri_span:
                                tri_id = t_id
                                break
                        if tri_id is None:
                            continue
                        arg_id = None
                        for ent_id, ent_tuple in ent_map.items():
                            if ent_tuple[1] == arg_type and ent_tuple[2] == arg_span:
                                arg_id = ent_id
                                break
                        if arg_id is None:
                            continue
                        if ">" in role:
                            raise ("error")
                        if role == "Cause":
                            continue
                        simple_relations.append((tri_type, tri_id, role, arg_id))
                temp_simple = sorted(simple_relations, key=lambda x: int(tri_map[x[1]][2].split(" ")[0]))
                simple_relations = {}
                for idx in range(len(temp_simple)):
                    simple_relations[f"E{idx + 1}"] = temp_simple[idx]
                    line = f"E{idx + 1}\t{temp_simple[idx][0]}:{temp_simple[idx][1]} {temp_simple[idx][2]}:{temp_simple[idx][3]}\n"
                    assert len(tri_map) > 0
                    f.write(line)

                max_event_count = len(simple_relations) + 1
                binding_relations = {}
                binding1_relations = set()
                binding2_relations = set()

                for relation in extracted_relation:
                    tri_span = relation.split("&&")[1].split(":")[0].replace("-", " ")
                    tri_type = relation.split(">>")[1].split("->")[0]
                    role = relation.split("->")[1].split("<-")[0].split("$")[0]
                    arg_type = relation.split("<-")[1].split("<<")[0]
                    arg_start, arg_end = relation.split(":")[-1].split("-")
                    arg_span = f"{arg_start} {arg_end}"
                    if tri_type in MThemes_events:
                        tri_id = None
                        for t_id, t_tuple in tri_map.items():
                            if t_tuple[1] == tri_type and t_tuple[2] == tri_span:
                                tri_id = t_id
                                break
                        if tri_id is None:
                            continue
                        arg_id = None
                        for ent_id, ent_tuple in ent_map.items():
                            if ent_tuple[1] == arg_type and ent_tuple[2] == arg_span:
                                arg_id = ent_id
                                break
                        if arg_id is None:
                            continue
                        if role.endswith("2"):
                            binding2_relations.add((tri_type, tri_id, role, arg_id))
                        elif role.startswith("Theme"):
                            binding1_relations.add((tri_type, tri_id, role, arg_id))

                used_binding_triggers = set()
                construct = set()
                have_tuple = set()
                remove_tuple = set()
                for values in binding2_relations:
                    tri_id, arg_id = values[1], values[3]
                    for values_one in binding1_relations:
                        if tri_id == values_one[1] and values_one[3] != arg_id and (
                                tri_id, arg_id, values_one[3]) not in have_tuple and (
                                tri_id, values_one[3], arg_id) not in have_tuple:
                            remove_tuple.add(values_one)
                            construct.add((values[0], tri_id, values_one[2], values_one[3], values[2], values[3]))
                            have_tuple.add((tri_id, arg_id, values_one[3]))
                            have_tuple.add((tri_id, values_one[3], arg_id))
                            used_binding_triggers.add(tri_id)
                construct = sorted(list(construct), key=lambda x: int(tri_map[x[1]][2].split(" ")[0]))

                if name in network_p and len(construct) > 0:
                    left_parts, absolute_parts, right_parts = set(), set(), set()
                    doc_network_p = network_p[name]
                    for node in doc_network_p:
                        if "Theme2" not in node:
                            continue
                        head_span = node.split("&&")[1].split(":")[0].replace("-", " ")
                        tail_span = node.split(":")[-1].replace("-", " ")
                        head_id, tail_id = None, None
                        for item in mix.values():
                            if head_span in item:
                                head_id = item[0]
                            if tail_span in item:
                                tail_id = item[0]
                        if head_id is not None and tail_id is None:
                            left_parts.add((head_id, tail_id))
                        if head_id is None and tail_id is not None:
                            right_parts.add((head_id, tail_id))
                        if head_id is not None and tail_id is not None:
                            absolute_parts.add((head_id, tail_id))
                    if len(absolute_parts) > 0:  # 提升了1.4： 59.47->60.81
                        new_construct = []
                        for item_ in construct:
                            pair = (item_[3], item_[5])
                            if pair in absolute_parts:
                                new_construct.append(item_)
                        construct = new_construct

                for item in construct:
                    item_ = list(item)
                    f.write(f"E{max_event_count}\t{item_[0]}:{item_[1]} {item_[2]}:{item_[3]} {item_[4]}:{item_[5]}\n")
                    binding_relations[f"E{max_event_count}"] = item
                    max_event_count += 1

                binding1_relations = sorted(list(binding1_relations), key=lambda x: int(tri_map[x[1]][2].split(" ")[0]))
                for values in binding1_relations:
                    if values in remove_tuple:
                        continue
                    used_binding_triggers.add(values[1])
                    f.write(f"E{max_event_count}\t{values[0]}:{values[1]} {values[2]}:{values[3]}\n")
                    binding_relations[f"E{max_event_count}"] = values
                    max_event_count += 1

                Theme_set, Cause_set = [], []
                for relation in extracted_relation:
                    tri_span = relation.split("&&")[1].split(":")[0].replace("-", " ")
                    tri_type = relation.split(">>")[1].split("->")[0]
                    role = relation.split("->")[1].split("<-")[0].split("$")[0]
                    arg_type = relation.split("<<")[0].split("<-")[1]
                    arg_start, arg_end = relation.split(":")[-1].split("-")
                    arg_span = f"{arg_start} {arg_end}"
                    if tri_type not in Nested_events:
                        continue
                    # 找ID
                    tri_id, arg_id = None, None
                    for key, value in mix.items():
                        if value[1] == tri_type and (
                                value[2] == tri_span or value[2].split()[0] == tri_span.split()[0]):
                            tri_id = key
                            break
                    if tri_id is None:
                        continue
                    for key, value in mix.items():
                        if value[1] == arg_type and (
                                value[2] == arg_span or value[2].split()[0] == arg_span.split()[0]):
                            arg_id = key
                            break
                    if arg_id is None:
                        continue
                    if tri_type in Nested_events:
                        if role == "Theme":
                            Theme_set.append((tri_type, tri_id, arg_id, arg_type, role))
                        elif role == "Cause":
                            Cause_set.append((tri_type, tri_id, arg_id, arg_type, role))
                # 对于嵌套事件，需要从底向上进行枚举, 这里定义实体为Leve0
                # 枚举: (level0,), (level0, level0)
                nested_set1 = []
                have_tuple = set()
                for theme_info in Theme_set:
                    if theme_info[3] == "Protein":
                        flag = True
                        for cause_info in Cause_set:
                            if cause_info[3] == "Protein" and theme_info[1] == cause_info[1] and \
                                    (theme_info[1], theme_info[2], cause_info[2]) not in have_tuple and \
                                    (theme_info[1], cause_info[2], theme_info[2]):
                                if ["Protein", "Protein"] not in structure[theme_info[0]]:
                                    continue
                                nested_set1.append(
                                    (theme_info[0], theme_info[1], theme_info[4], theme_info[2], cause_info[4],
                                     cause_info[2]))
                                have_tuple.add((theme_info[1], theme_info[2], cause_info[2]))
                                have_tuple.add((theme_info[1], cause_info[2], theme_info[2]))
                                flag = False
                        if flag and (theme_info[1], theme_info[2]) not in have_tuple and (
                                theme_info[2], theme_info[1]) not in have_tuple:
                            if ["Protein"] not in structure[theme_info[0]]:
                                continue
                            nested_set1.append((theme_info[0], theme_info[1], theme_info[4], theme_info[2]))
                            have_tuple.add((theme_info[1], theme_info[2]))
                            have_tuple.add((theme_info[2], theme_info[1]))
                nested_level1 = sorted(list(nested_set1), key=lambda x: int(tri_map[x[1]][2].split(" ")[0]))
                level1_tc_idx = [i for i, item_ in enumerate(nested_level1) if len(item_) > 4]
                if name in network_p and len(level1_tc_idx) > 0:
                    left_parts, absolute_parts, right_parts = set(), set(), set()
                    doc_network_p = network_p[name]
                    for node in doc_network_p:
                        if "Theme2" in node:
                            continue
                        head_span = node.split("&&")[1].split(":")[0].replace("-", " ")
                        tail_span = node.split(":")[-1].replace("-", " ")
                        head_id, tail_id = None, None
                        for item in mix.values():
                            if head_span in item:
                                head_id = item[0]
                            if tail_span in item:
                                tail_id = item[0]
                        if head_id is not None and tail_id is None:
                            left_parts.add((head_id, tail_id))
                        if head_id is None and tail_id is not None:
                            right_parts.add((head_id, tail_id))
                        if head_id is not None and tail_id is not None:
                            absolute_parts.add((head_id, tail_id))
                    if len(absolute_parts) > 0:  # 提升了1.4： 59.47->60.81
                        new_nested_level1 = []
                        for idx, item_ in enumerate(nested_level1):
                            if idx in level1_tc_idx:
                                pair = (item_[3], item_[5])
                                if pair in absolute_parts:
                                    new_nested_level1.append(item_)
                            else:
                                new_nested_level1.append(item_)
                        nested_level1 = new_nested_level1
                event_level1 = {}
                event_level1.update(simple_relations)
                event_level1.update(binding_relations)
                for item in nested_level1:
                    if len(item) == 6:
                        line = f"E{max_event_count}\t{item[0]}:{item[1]} {item[2]}:{item[3]} {item[4]}:{item[5]}\n"
                        event_level1[f"E{max_event_count}"] = item
                        f.write(line)
                    else:
                        line = f"E{max_event_count}\t{item[0]}:{item[1]} {item[2]}:{item[3]}\n"
                        event_level1[f"E{max_event_count}"] = item
                        f.write(line)
                    max_event_count += 1
                nested_set2 = []
                # 枚举: (level1,), (level0, level1), (level1, level0), (level1, level1)
                for theme_info in Theme_set:
                    if theme_info[3] != "Protein":
                        # 查找事件修改theme_info的值
                        theme_e_ids = []
                        for e_id, e_tuple in event_level1.items():
                            if e_tuple[1] == theme_info[2]:
                                theme_e_ids.append(e_id)
                        if len(theme_e_ids) == 0:
                            continue
                        for theme_e_id in theme_e_ids:
                            theme_info_ = list(theme_info)
                            theme_info_[2] = theme_e_id
                            flag = True
                            for cause_info in Cause_set:
                                if cause_info[3] != "Protein":  # (level1, level1)
                                    cause_e_ids = []
                                    for e_id, e_tuple in event_level1.items():
                                        if e_tuple[1] == cause_info[2]:
                                            cause_e_ids.append(e_id)
                                    if len(cause_e_ids) == 0:
                                        continue
                                    for cause_e_id in cause_e_ids:
                                        type_pair = sorted([theme_info[3], cause_info[3]])
                                        if type_pair not in structure[theme_info[0]]:
                                            continue
                                        cause_info_ = list(cause_info)
                                        cause_info_[2] = cause_e_id
                                        if theme_info_[1] == cause_info_[1] and \
                                                (theme_info_[1], theme_info_[2], cause_info_[2]) not in have_tuple and \
                                                (theme_info_[1], cause_info_[2], theme_info_[2]):
                                            nested_set2.append(
                                                (theme_info_[0], theme_info_[1], theme_info_[4], theme_info_[2],
                                                 cause_info_[4],
                                                 cause_info_[2]))
                                            have_tuple.add((theme_info_[1], theme_info_[2], cause_info_[2]))
                                            have_tuple.add((theme_info_[1], cause_info_[2], theme_info_[2]))
                                            flag = False
                                else:
                                    if theme_info_[1] == cause_info[1] and \
                                            (theme_info_[1], theme_info_[2], cause_info[2]) not in have_tuple and \
                                            (theme_info_[1], cause_info[2], theme_info_[2]):  # (level1, level0)
                                        nested_set2.append(
                                            (theme_info_[0], theme_info_[1], theme_info_[4], theme_info_[2],
                                             cause_info[4],
                                             cause_info[2]))
                                        type_pair = sorted([theme_info[3], cause_info[3]])
                                        if type_pair not in structure[theme_info[0]]:
                                            continue
                                        have_tuple.add((theme_info_[1], theme_info_[2], cause_info[2]))
                                        have_tuple.add((theme_info_[1], cause_info[2], theme_info_[2]))
                                        flag = False
                            if flag and (theme_info_[1], theme_info_[2]) not in have_tuple and (
                                    theme_info_[2], theme_info_[1]) not in have_tuple:  # (level1,)
                                type_pair = sorted([theme_info[3]])
                                if type_pair not in structure[theme_info[0]]:
                                    continue
                                nested_set2.append((theme_info_[0], theme_info_[1], theme_info_[4], theme_info_[2]))
                                have_tuple.add((theme_info_[1], theme_info_[2]))
                                have_tuple.add((theme_info_[2], theme_info_[1]))

                    else:
                        for cause_info in Cause_set:
                            if cause_info[3] != "Protein":  # (level0, level1)
                                cause_e_ids = []
                                for e_id, e_tuple in event_level1.items():
                                    if e_tuple[1] == cause_info[2]:
                                        cause_e_ids.append(e_id)
                                if len(cause_e_ids) == 0:
                                    continue
                                for cause_e_id in cause_e_ids:
                                    cause_info_ = list(cause_info)
                                    cause_info_[2] = cause_e_id
                                    type_pair = sorted([theme_info[3], cause_info[3]])
                                    if type_pair not in structure[theme_info[0]]:
                                        continue
                                    if theme_info[1] == cause_info_[1] and \
                                            (theme_info[1], theme_info[2], cause_info_[2]) not in have_tuple and \
                                            (theme_info[1], cause_info_[2], theme_info[2]):
                                        nested_set2.append(
                                            (theme_info[0], theme_info[1], theme_info[4], theme_info[2], cause_info_[4],
                                             cause_info_[2]))
                                        have_tuple.add((theme_info[1], theme_info[2], cause_info_[2]))
                                        have_tuple.add((theme_info[1], cause_info_[2], theme_info[2]))

                nested_level2 = sorted(list(nested_set2), key=lambda x: int(tri_map[x[1]][2].split(" ")[0]))
                level2_tc_idx = [i for i, item_ in enumerate(nested_level2) if len(item_) > 4]
                if name in network_p and len(level2_tc_idx) > 0:
                    left_parts, absolute_parts, right_parts = set(), set(), set()
                    doc_network_p = network_p[name]
                    for node in doc_network_p:
                        if "Theme2" in node:
                            continue
                        head_span = node.split("&&")[1].split(":")[0].replace("-", " ")
                        tail_span = node.split(":")[-1].replace("-", " ")
                        head_id, tail_id = None, None
                        for item in mix.values():
                            if head_span in item:
                                head_id = item[0]
                            if tail_span in item:
                                tail_id = item[0]
                        if head_id is not None and tail_id is None:
                            left_parts.add((head_id, tail_id))
                        if head_id is None and tail_id is not None:
                            right_parts.add((head_id, tail_id))
                        if head_id is not None and tail_id is not None:
                            absolute_parts.add((head_id, tail_id))
                    if len(absolute_parts) > 0:  # 提升了1.4： 59.47->60.81
                        new_nested_level2 = []
                        for idx, item_ in enumerate(nested_level2):
                            if idx in level2_tc_idx:
                                head_idx, tail_idx = item_[3], item_[5]
                                if head_idx.startswith("E"):
                                    head_idx = event_level1[head_idx][1]
                                if tail_idx.startswith("E"):
                                    tail_idx = event_level1[tail_idx][1]
                                pair = (head_idx, tail_idx)
                                if pair in absolute_parts:
                                    new_nested_level2.append(item_)
                            else:
                                new_nested_level2.append(item_)
                        nested_level2 = new_nested_level2
                event_level2 = {}
                for item in nested_level2:
                    if len(item) == 6:
                        line = f"E{max_event_count}\t{item[0]}:{item[1]} {item[2]}:{item[3]} {item[4]}:{item[5]}\n"
                        event_level2[f"E{max_event_count}"] = item
                        f.write(line)
                    else:
                        line = f"E{max_event_count}\t{item[0]}:{item[1]} {item[2]}:{item[3]}\n"
                        event_level2[f"E{max_event_count}"] = item
                        f.write(line)
                    max_event_count += 1
                # 枚举: (level2,), (level0, level2), (level2, level0), (level1, level2), (level2, level1), (level2, level2)
                # STEP-1: 枚举(level2,), (level2, level0), (level2, level1), (level0, level2)
                event_level12 = {}
                event_level12.update(event_level1)
                event_level12.update(event_level2)
                nested_set3 = []
                for theme_info in Theme_set:
                    if theme_info[3] != "Protein":
                        # 查找事件修改theme_info的值
                        theme_e_ids = []
                        for e_id, e_tuple in event_level2.items():
                            if e_tuple[1] == theme_info[2]:
                                theme_e_ids.append(e_id)
                        if len(theme_e_ids) == 0:
                            continue
                        for theme_e_id in theme_e_ids:
                            theme_info_ = list(theme_info)
                            theme_info_[2] = theme_e_id
                            flag = True
                            for cause_info in Cause_set:
                                if cause_info[3] != "Protein":  # (level2, level1)
                                    cause_e_ids = []
                                    for e_id, e_tuple in event_level1.items():
                                        if e_tuple[1] == cause_info[2]:
                                            cause_e_ids.append(e_id)
                                    if len(cause_e_ids) == 0:
                                        continue
                                    for cause_e_id in cause_e_ids:
                                        type_pair = sorted([theme_info[3], cause_info[3]])
                                        if type_pair not in structure[theme_info[0]]:
                                            continue
                                        cause_info_ = list(cause_info)
                                        cause_info_[2] = cause_e_id
                                        if theme_info_[1] == cause_info_[1] and \
                                                (theme_info_[1], theme_info_[2], cause_info_[2]) not in have_tuple and \
                                                (theme_info_[1], cause_info_[2], theme_info_[2]):
                                            nested_set3.append(
                                                (theme_info_[0], theme_info_[1], theme_info_[4], theme_info_[2],
                                                 cause_info_[4],
                                                 cause_info_[2]))
                                            have_tuple.add((theme_info_[1], theme_info_[2], cause_info_[2]))
                                            have_tuple.add((theme_info_[1], cause_info_[2], theme_info_[2]))
                                            flag = False
                                else:
                                    if theme_info_[1] == cause_info[1] and \
                                            (theme_info_[1], theme_info_[2], cause_info[2]) not in have_tuple and \
                                            (theme_info_[1], cause_info[2], theme_info_[2]):  # (level2, level0)
                                        type_pair = sorted([theme_info[3], cause_info[3]])
                                        if type_pair not in structure[theme_info[0]]:
                                            continue
                                        nested_set3.append(
                                            (theme_info_[0], theme_info_[1], theme_info_[4], theme_info_[2],
                                             cause_info[4],
                                             cause_info[2]))
                                        have_tuple.add((theme_info_[1], theme_info_[2], cause_info[2]))
                                        have_tuple.add((theme_info_[1], cause_info[2], theme_info_[2]))
                                        flag = False
                        if flag and (theme_info_[1], theme_info_[2]) not in have_tuple and (
                                theme_info_[2], theme_info_[1]) not in have_tuple:  # (level2,)
                            type_pair = sorted([theme_info[3]])
                            if type_pair not in structure[theme_info[0]]:
                                continue
                            nested_set3.append((theme_info_[0], theme_info_[1], theme_info_[4], theme_info_[2]))
                            have_tuple.add((theme_info_[1], theme_info_[2]))
                            have_tuple.add((theme_info_[2], theme_info_[1]))

                    else:
                        for cause_info in Cause_set:
                            if cause_info[3] != "Protein":  # (level0, level2)
                                cause_e_ids = []
                                for e_id, e_tuple in event_level2.items():
                                    if e_tuple[1] == cause_info[2]:
                                        cause_e_ids.append(e_id)
                                if len(cause_e_ids) == 0:
                                    continue
                                for cause_e_id in cause_e_ids:
                                    type_pair = sorted([theme_info[3], cause_info[3]])
                                    if type_pair not in structure[theme_info[0]]:
                                        continue
                                    cause_info_ = list(cause_info)
                                    cause_info_[2] = cause_e_id
                                    if theme_info[1] == cause_info_[1] and \
                                            (theme_info[1], theme_info[2], cause_info_[2]) not in have_tuple and \
                                            (theme_info[1], cause_info_[2], theme_info[2]):
                                        nested_set3.append(
                                            (theme_info[0], theme_info[1], theme_info[4], theme_info[2], cause_info_[4],
                                             cause_info_[2]))
                                        have_tuple.add((theme_info[1], theme_info[2], cause_info_[2]))
                                        have_tuple.add((theme_info[1], cause_info_[2], theme_info[2]))
                # STEP-2 枚举 (level1, level2)
                for theme_info in Theme_set:
                    if theme_info[3] != "Protein":
                        # 查找事件修改theme_info的值
                        theme_e_ids = []
                        for e_id, e_tuple in event_level1.items():
                            if e_tuple[1] == theme_info[2]:
                                theme_e_ids.append(e_id)
                        if len(theme_e_ids) == 0:
                            continue
                        for theme_e_id in theme_e_ids:
                            theme_info_ = list(theme_info)
                            theme_info_[2] = theme_e_id
                            for cause_info in Cause_set:
                                if cause_info[3] != "Protein":  # (level1, level2)
                                    cause_e_ids = []
                                    for e_id, e_tuple in event_level2.items():
                                        if e_tuple[1] == cause_info[2]:
                                            cause_e_ids.append(e_id)
                                    if len(cause_e_ids) == 0:
                                        continue
                                    for cause_e_id in cause_e_ids:
                                        type_pair = sorted([theme_info[3], cause_info[3]])
                                        if type_pair not in structure[theme_info[0]]:
                                            continue
                                        cause_info_ = list(cause_info)
                                        cause_info_[2] = cause_e_id
                                        if theme_info_[1] == cause_info_[1] and \
                                                (theme_info_[1], theme_info_[2], cause_info_[2]) not in have_tuple and \
                                                (theme_info_[1], cause_info_[2], theme_info_[2]):
                                            nested_set3.append(
                                                (theme_info_[0], theme_info_[1], theme_info_[4], theme_info_[2],
                                                 cause_info_[4],
                                                 cause_info_[2]))
                                            have_tuple.add((theme_info_[1], theme_info_[2], cause_info_[2]))
                                            have_tuple.add((theme_info_[1], cause_info_[2], theme_info_[2]))
                # STEP-4 枚举(level2, level2)
                for theme_info in Theme_set:
                    if theme_info[3] != "Protein":
                        # 查找事件修改theme_info的值
                        theme_e_ids = []
                        for e_id, e_tuple in event_level2.items():
                            if e_tuple[1] == theme_info[2]:
                                theme_e_ids.append(e_id)
                        if len(theme_e_ids) == 0:
                            continue
                        for theme_e_id in theme_e_ids:
                            theme_info_ = list(theme_info)
                            theme_info_[2] = theme_e_id
                            for cause_info in Cause_set:
                                if cause_info[3] != "Protein":  # (level2, level2)
                                    cause_e_ids = []
                                    for e_id, e_tuple in event_level2.items():
                                        if e_tuple[1] == cause_info[2]:
                                            cause_e_ids.append(e_id)
                                            # break
                                    if len(cause_e_ids) == 0:
                                        continue
                                    for cause_e_id in cause_e_ids:
                                        type_pair = sorted([theme_info[3], cause_info[3]])
                                        if type_pair not in structure[theme_info[0]]:
                                            continue

                                        cause_info_ = list(cause_info)
                                        cause_info_[2] = cause_e_id
                                        if theme_info_[1] == cause_info_[1] and \
                                                (theme_info_[1], theme_info_[2], cause_info_[2]) not in have_tuple and \
                                                (theme_info_[1], cause_info_[2], theme_info_[2]):
                                            nested_set3.append(
                                                (theme_info_[0], theme_info_[1], theme_info_[4], theme_info_[2],
                                                 cause_info_[4],
                                                 cause_info_[2]))
                                            have_tuple.add((theme_info_[1], theme_info_[2], cause_info_[2]))
                                            have_tuple.add((theme_info_[1], cause_info_[2], theme_info_[2]))
                nested_level3 = sorted(list(nested_set3), key=lambda x: int(tri_map[x[1]][2].split(" ")[0]))
                level3_tc_idx = [i for i, item_ in enumerate(nested_level3) if len(item_) > 4]
                if name in network_p and len(level3_tc_idx) > 0:
                    left_parts, absolute_parts, right_parts = set(), set(), set()
                    doc_network_p = network_p[name]
                    for node in doc_network_p:
                        if "Theme2" in node:
                            continue
                        head_span = node.split("&&")[1].split(":")[0].replace("-", " ")
                        tail_span = node.split(":")[-1].replace("-", " ")
                        head_id, tail_id = None, None
                        for item in mix.values():
                            if head_span in item:
                                head_id = item[0]
                            if tail_span in item:
                                tail_id = item[0]
                        if head_id is not None and tail_id is None:
                            left_parts.add((head_id, tail_id))
                        if head_id is None and tail_id is not None:
                            right_parts.add((head_id, tail_id))
                        if head_id is not None and tail_id is not None:
                            absolute_parts.add((head_id, tail_id))
                    if len(absolute_parts) > 0:  # 提升了1.4： 59.47->60.81
                        new_nested_level3 = []
                        for idx, item_ in enumerate(nested_level3):
                            if idx in level3_tc_idx:
                                head_idx, tail_idx = item_[3], item_[5]
                                if head_idx.startswith("E"):
                                    head_idx = event_level12[head_idx][1]
                                if tail_idx.startswith("E"):
                                    tail_idx = event_level12[tail_idx][1]
                                pair = (head_idx, tail_idx)
                                if pair in absolute_parts:
                                    new_nested_level3.append(item_)
                            else:
                                new_nested_level3.append(item_)
                        nested_level3 = new_nested_level3
                event_level3 = {}
                for item in nested_level3:
                    if len(item) == 6:
                        line = f"E{max_event_count}\t{item[0]}:{item[1]} {item[2]}:{item[3]} {item[4]}:{item[5]}\n"
                        event_level3[f"E{max_event_count}"] = item
                        f.write(line)
                    else:
                        line = f"E{max_event_count}\t{item[0]}:{item[1]} {item[2]}:{item[3]}\n"
                        event_level3[f"E{max_event_count}"] = item
                        f.write(line)
                    max_event_count += 1


    # normalization
    # normalize_cmd = f'./evaluation/GE13/tools/a2-normalize.pl -g ./sources/GE13/devel/ -u ./experiments/GE13/{mode}/a2/*.a2'
    normalize_cmd = f'./evaluation/GE13/tools/a2-normalize.pl -g ./sources/GE13/{mode}/ -u ./experiments/GE13/{mode}/a2/*.a2'
    p = subprocess.Popen(normalize_cmd, stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()

    # eval
    eval_cmd = f'./evaluation/GE13/tools/a2-evaluate.pl -g ./sources/GE13/{mode}/ -t1 -sp ./experiments/GE13/{mode}/a2/*.a2'
    p = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE, shell=True)
    output, err = p.communicate()
    output = output.decode('utf-8')
    # output = output.decode('gbk')
    if logger is not None:
        logger.info(output)
    else:
        print(output)
    end2end_score = float(re.findall(r"([\d+\.]+)\n", output)[-1]) / 100
    if mode == "test":
        eval_cmd = f'tar -zcvf filename.tar.gz ./experiments/{model_name}/{mode}/*'
        p = subprocess.Popen(eval_cmd, stdout=subprocess.PIPE, shell=True)
    return end2end_score


if __name__ == '__main__':
    root_path = "experiments/GE13"
    entity_g = json.load(open(os.path.join(root_path, "devel_entity_g.json"), encoding="utf-8", mode="r"))
    trigger_p = json.load(open(os.path.join(root_path, "devel_trigger_p.json"), encoding="utf-8", mode="r"))
    relation_p = json.load(open(os.path.join(root_path, "devel_relation_p.json"), encoding="utf-8", mode="r"))
    network_p = json.load(open(os.path.join(root_path, "devel_network_p.json"), encoding="utf-8", mode="r"))
    aux = [trigger_p, None, relation_p, None, network_p, None, entity_g]
    source_path = "sources/GE13"
    mode = "devel"
    save_path = "experiments/GE13"
    merge(aux=aux, save_path=save_path, source_path=source_path, mode=mode)
