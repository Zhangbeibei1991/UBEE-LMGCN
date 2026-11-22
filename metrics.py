import torch
import numpy as np
import cupy as cp
from tqdm import tqdm

Theme_events = ["Gene_expression", "Transcription", "Localization", "Protein_catabolism"]
Theme_Cause_events = ["Regulation", "Negative_regulation", "Positive_regulation", "Phosphorylation", "Ubiquitination",
                      "Deacetylation", "Protein_modification", "Acetylation"]
MThemes_events = ["Binding"]
Nest_events = ["Regulation", "Positive_regulation", "Negative_regulation"]
Others_events = ["Anaphora", "Entity"]
all_events = Theme_events + Theme_Cause_events + MThemes_events + Others_events
all_roles = ["Theme", "Theme2", "Theme3", "Cause"]
all_edges = ["Theme$Cause", "Theme$Theme2"]


def performance_report(y_true, y_pred):
    pred_num = len(y_pred)
    gold_num = len(y_true)
    corr_num = len(y_true & y_pred)
    P = corr_num / (pred_num + 1e-10)
    R = corr_num / (gold_num + 1e-10)
    F = 2 * P * R / (P + R + 1e-10)
    return P, R, F


def span_generation(logits, gold, id2lab, info=None, inner_dict=None, logger=None):
    logits = logits.cpu().numpy()
    gold = gold.cpu().numpy()
    pred = {}
    true = {}
    pred_set, true_set = [], []
    sent_idx = info["sent_idx"]
    tri_std, rel_std, net_std, entities = inner_dict
    for sent_id in sent_idx:
        doc_id = sent_id.split(":s")[0]
        entity = info["entity"][sent_id]
        sent_start = info["sent_start"][sent_id]
        org_text = info["org_text"][sent_id]
        tokens = info["token"][sent_id]
        if doc_id not in entities:
            entities[doc_id] = set()

        for ent in entity:
            ent_id = ent[0]
            ent_start, ent_end = ent[2], ent[3]
            ent_txt = ent[-1]
            ent_type = ent[1]
            entities[doc_id].add(
                (ent_id, ent_type, ent_start, ent_end, ent_txt, org_text, " ".join(tokens), sent_start, "Entity"))

    for b, l, start, end in zip(*np.where(logits > 0)):
        doc_id = sent_idx[b].split(":s")[0]
        sent_id = sent_idx[b]
        char_map = info["char_map"][sent_idx[b]]
        sent_start = info["sent_start"][sent_idx[b]]
        org_text = info["org_text"][sent_idx[b]]
        tokens = info["token"][sent_idx[b]]
        if doc_id not in pred:
            pred[doc_id] = set()
        if doc_id not in tri_std:
            tri_std[doc_id] = {"pred": [], "gold": []}
        if doc_id not in rel_std:
            rel_std[doc_id] = {"pred": [], "gold": []}
        if doc_id not in net_std:
            net_std[doc_id] = {"pred": [], "gold": []}

        if id2lab[l] in all_events:
            char_start, char_end = char_map[start][0], char_map[end][-1]
            if char_start > char_end:
                continue
            line = (doc_id, sent_id, id2lab[l], char_start + sent_start, char_end + sent_start,
                    org_text[char_start:char_end], org_text, " ".join(tokens), sent_start, "Trigger")
            pred[doc_id].add(line)
            pred_set.append(line)
            tri_std[doc_id]["pred"].append(line)
        else:
            char_start, char_end = char_map[start][0], char_map[end][0]
            if "_rev" in id2lab[l]:
                indicator = "Relation"
                if "$" in id2lab[l]:
                    indicator = "Network"
                line = (doc_id, sent_id, id2lab[l].split("_")[0], char_end + sent_start, char_start + sent_start, None,
                        org_text, " ".join(tokens), sent_start, indicator)
                pred[doc_id].add(line)
                pred_set.append(line)
                if "$" not in id2lab[l]:
                    rel_std[doc_id]["pred"].append(line)
                if "$" in id2lab[l]:
                    net_std[doc_id]["pred"].append(line)
            else:
                indicator = "Relation"
                if "$" in id2lab[l]:
                    indicator = "Network"
                line = (doc_id, sent_id, id2lab[l].split("_")[0], char_start + sent_start, char_end + sent_start, None,
                        org_text, " ".join(tokens), sent_start, indicator)
                pred[doc_id].add(line)
                pred_set.append(line)
                if "$" not in id2lab[l]:
                    rel_std[doc_id]["pred"].append(line)
                if "$" in id2lab[l]:
                    net_std[doc_id]["pred"].append(line)

    for b, l, start, end in zip(*np.where(gold > 0)):
        doc_id = sent_idx[b].split(":s")[0]
        sent_id = sent_idx[b]
        char_map = info["char_map"][sent_idx[b]]
        sent_start = info["sent_start"][sent_idx[b]]
        org_text = info["org_text"][sent_idx[b]]
        tokens = info["token"][sent_idx[b]]

        if doc_id not in true:
            true[doc_id] = set()
        if doc_id not in tri_std:
            tri_std[doc_id] = {"pred": [], "gold": []}
        if doc_id not in rel_std:
            rel_std[doc_id] = {"pred": [], "gold": []}
        if doc_id not in net_std:
            net_std[doc_id] = {"pred": [], "gold": []}
        if id2lab[l] in all_events:
            char_start, char_end = char_map[start][0], char_map[end][-1]
            if char_start > char_end:
                continue

            line = (doc_id, sent_id, id2lab[l], char_start + sent_start, char_end + sent_start,
                    org_text[char_start:char_end], org_text, " ".join(tokens), sent_start, "Trigger")
            true[doc_id].add(line)
            true_set.append(line)
            if line not in tri_std[doc_id]["gold"]:
                tri_std[doc_id]["gold"].append(line)
        else:
            char_start, char_end = char_map[start][0], char_map[end][0]
            if "_rev" in id2lab[l]:
                indicator = "Relation"
                if "$" in id2lab[l]:
                    indicator = "Network"
                line = (doc_id, sent_id, id2lab[l].split("_")[0], char_end + sent_start, char_start + sent_start, None,
                        org_text, " ".join(tokens), sent_start, indicator)
                true[doc_id].add(line)
                true_set.append(line)
                if "$" not in id2lab[l] and line not in rel_std[doc_id]["gold"]:
                    rel_std[doc_id]["gold"].append(line)
                if "$" in id2lab[l] and line not in net_std[doc_id]["gold"]:
                    net_std[doc_id]["gold"].append(line)
            else:
                indicator = "Relation"
                if "$" in id2lab[l]:
                    indicator = "Network"
                line = (doc_id, sent_id, id2lab[l].split("_")[0], char_start + sent_start, char_end + sent_start, None,
                        org_text, " ".join(tokens), sent_start, indicator)
                true[doc_id].add(line)
                true_set.append(line)
                if "$" not in id2lab[l] and line not in rel_std[doc_id]["gold"]:
                    rel_std[doc_id]["gold"].append(line)
                if "$" in id2lab[l] and line not in net_std[doc_id]["gold"]:
                    net_std[doc_id]["gold"].append(line)
    return pred, true, pred_set, true_set

def detail_scores(aux):
    doc_spans, tri_std, rel_std, net_std, entities, prior, logger, source_path = aux
    for doc_id, doc_tris in tri_std.items():
        pred_items = doc_tris["pred"]
        gold_items = doc_tris["gold"]
        sent_spans = doc_spans[doc_id]
        doc_tri_set = set()
        # aaa = set([(item[0], item[2], item[3], item[4], item[5]) for item in pred_items])
        for pred_item in pred_items:
            pred_start, pred_end, sent_start = pred_item[3], pred_item[4], pred_item[-2]
            sent_end = sent_start + len(pred_item[-4])
            single_start, single_end = sent_spans[f"{sent_start}-{sent_end}"]
            if single_start <= pred_start <= pred_end <= single_end:
                doc_tri_set.add((pred_item[0], pred_item[2], pred_item[3], pred_item[4], pred_item[5]))
        tri_std[doc_id]["pred"] = doc_tri_set
        doc_tri_set = set()
        for gold_item in gold_items:
            gold_start, gold_end, sent_start = gold_item[3], gold_item[4], gold_item[-2]
            sent_end = sent_start + len(gold_item[-4])
            single_start, single_end = sent_spans[f"{sent_start}-{sent_end}"]
            if single_start <= gold_start <= gold_end <= single_end:
                doc_tri_set.add((gold_item[0], gold_item[2], gold_item[3], gold_item[4], gold_item[5]))
        tri_std[doc_id]["gold"] = doc_tri_set

    pred, gold = set(), set()
    trigger_g, trigger_p = {}, {}
    relation_g, relation_p = {}, {}
    network_p, network_g = {}, {}
    mix1, mix2 = {}, {}
    for doc_id, ent_items in entities.items():
        if doc_id not in mix1:
            mix1[doc_id] = set()
            mix2[doc_id] = set()
        for ent_item in ent_items:
            mix1[doc_id].add((ent_item[0], ent_item[1], ent_item[2], ent_item[3], ent_item[4]))
            mix2[doc_id].add((ent_item[0], ent_item[1], ent_item[2], ent_item[3], ent_item[4]))
        if doc_id in tri_std:
            mix1[doc_id].update(tri_std[doc_id]["pred"])
            mix2[doc_id].update(tri_std[doc_id]["gold"])
    lost_count, total_count = 0, 0
    for doc_id, rel_items in rel_std.items():
        arg_pred, arg_gold = set(), set()
        for rel_item in rel_items["pred"]:
            total_count += 1
            _, _, role, tri_start, arg_start, _ = rel_item[:6]
            trigger_part = None
            if doc_id in tri_std:
                pred_tris = tri_std[doc_id]["pred"]
                for pred_tri in pred_tris:
                    _, pred_t_type, pred_t_start, pred_t_end, pred_t_text = pred_tri
                    if pred_t_start <= tri_start < pred_t_end:
                        trigger_part = pred_tri
                        break
                else:
                    doc_txt = open(f"{source_path}/{doc_id}.txt", encoding="utf-8", mode="r").read()
                    tri_txt = doc_txt[tri_start:].split()[0]
                    tri_end = tri_start + len(tri_txt)
                    if tri_txt.lower() in prior:
                        trigger_part = (doc_id, prior[tri_txt.lower()][0][0], tri_start, tri_end, tri_txt)
                        build_type = prior[tri_txt.lower()][0][0]
                        tri_std[doc_id]["pred"].add((doc_id, build_type, tri_start, tri_end, tri_txt))

            argument_part = None
            if doc_id in mix1:
                pred_items = sorted(list(mix1[doc_id]))
                for pred_item in pred_items:
                    _, pred_a_type, pred_a_start, pred_a_end, pred_a_text = pred_item
                    if pred_a_start <= arg_start < pred_a_end:
                        argument_part = pred_item
                        break
                else:
                    doc_txt = open(f"{source_path}/{doc_id}.txt", encoding="utf-8", mode="r").read()
                    arg_txt = doc_txt[arg_start:].split()[0]
                    arg_end = arg_start + len(arg_txt)
                    if arg_txt.lower() in prior:
                        argument_part = (doc_id, prior[arg_txt.lower()][0][0], arg_start, arg_end, arg_txt)
                        build_type = prior[arg_txt.lower()][0][0]
                        tri_std[doc_id]["pred"].add((doc_id, build_type, arg_start, arg_end, arg_txt))

            if trigger_part is None or argument_part is None:
                lost_count += 1
            else:
                arg_pred.add((doc_id, trigger_part[1], trigger_part[2], trigger_part[3], trigger_part[4],
                              argument_part[1], argument_part[2], argument_part[3], argument_part[4], role))
        rel_std[doc_id]["pred"] = arg_pred

        for rel_item in rel_items["gold"]:
            _, sid, role, tri_start, arg_start, _ = rel_item[:6]
            trigger_part = None
            if doc_id in tri_std:
                pred_tris = tri_std[doc_id]["gold"]
                for pred_tri in pred_tris:
                    _, pred_t_type, pred_t_start, pred_t_end, pred_t_text = pred_tri
                    if pred_t_start <= tri_start < pred_t_end:
                        trigger_part = pred_tri
                        break
                else:
                    raise ("error")

            argument_part = None
            if doc_id in mix2:
                pred_items = sorted(list(mix2[doc_id]))
                for pred_item in pred_items:
                    _, pred_a_type, pred_a_start, pred_a_end, pred_a_text = pred_item
                    if pred_a_start <= arg_start < pred_a_end:
                        argument_part = pred_item
                        break
                else:
                    raise ("error")
            assert doc_id in entities
            arg_gold.add((doc_id, trigger_part[1], trigger_part[2], trigger_part[3], trigger_part[4],
                          argument_part[1], argument_part[2], argument_part[3], argument_part[4], role))

            rel_std[doc_id]["gold"] = arg_gold
    logger.info(f"wreck rate (relation): {round(lost_count / (total_count + 1e-10), 4)}({lost_count}|{total_count})")
    for doc_id, rel_items in rel_std.items():
        gold.update(rel_items["gold"])
        pred.update(rel_items["pred"])
        for rel_item in rel_items["gold"]:
            if doc_id not in relation_g:
                relation_g[doc_id] = []
            rel_item_str = f"{rel_item[0]}&&{rel_item[2]}-{rel_item[3]}:{rel_item[4]}>>{rel_item[1]}->{rel_item[-1]}<-{rel_item[5]}<<{rel_item[8]}:{rel_item[6]}-{rel_item[7]}"
            if rel_item_str not in relation_g[doc_id]:
                relation_g[doc_id].append(rel_item_str)
        for rel_item in rel_items["pred"]:
            if doc_id not in relation_p:
                relation_p[doc_id] = []
            rel_item_str = f"{rel_item[0]}&&{rel_item[2]}-{rel_item[3]}:{rel_item[4]}>>{rel_item[1]}->{rel_item[-1]}<-{rel_item[5]}<<{rel_item[8]}:{rel_item[6]}-{rel_item[7]}"
            if rel_item_str not in relation_p[doc_id]:
                relation_p[doc_id].append(rel_item_str)
    pt, rt, ft = performance_report(y_true=gold, y_pred=pred)
    logger.info(f"devel metric".center(30, "*"))
    logger.info("precision (Relation): {0:>.4%}".format(pt))
    logger.info("recall (Relation): {0:>.4%}".format(rt))
    logger.info("f1 (Relation): {0:>.4%}".format(ft))  # 65.8761%|65.6876% -> gold: 3591, pred: 3160
    logger.info(f"*" * 30)

    binding1_theme_p, binding1_theme_g = set(), set()
    binding2_theme_p, binding2_theme_g = set(), set()
    for doc_id, rel_items in rel_std.items():
        g_set, p_set = rel_items["gold"], rel_items["pred"]
        for item in p_set:
            if "Binding" == item[1] and "Theme" == item[-1]:
                binding1_theme_p.add(item)
        for item in g_set:
            if "Binding" == item[1] and "Theme" == item[-1]:
                binding1_theme_g.add(item)

        for item in p_set:
            if "Binding" == item[1] and "Theme2" == item[-1]:
                binding2_theme_p.add(item)
        for item in g_set:
            if "Binding" == item[1] and "Theme2" == item[-1]:
                binding2_theme_g.add(item)

    corr_num = len(binding1_theme_p & binding1_theme_g)
    pred_num = len(binding1_theme_p)
    gold_num = len(binding1_theme_g)
    P = corr_num / (pred_num + 1e-10)
    R = corr_num / (gold_num + 1e-10)
    F = 2 * P * R / (P + R + 1e-10)
    logger.info(f"+" * 30)
    logger.info("precision (Binding-Theme): {0:>.4%}".format(P))
    logger.info("recall (Binding-Theme): {0:>.4%}".format(R))
    logger.info("f1 (Binding-Theme): {0:>.4%}".format(F))
    logger.info(f"+" * 30)
    corr_num = len(binding2_theme_p & binding2_theme_g)
    pred_num = len(binding2_theme_p)
    gold_num = len(binding2_theme_g)
    P = corr_num / (pred_num + 1e-10)
    R = corr_num / (gold_num + 1e-10)
    F = 2 * P * R / (P + R + 1e-10)
    logger.info(f"+" * 30)
    logger.info("precision (Binding-Theme2): {0:>.4%}".format(P))
    logger.info("recall (Binding-Theme2): {0:>.4%}".format(R))
    logger.info("f1 (Binding-Theme2): {0:>.4%}".format(F))
    logger.info(f"+" * 30)

    pos_reg_p, pos_reg_g = set(), set()
    neg_reg_p, neg_reg_g = set(), set()
    reg_p, reg_g = set(), set()
    for doc_id, rel_items in rel_std.items():
        g_set, p_set = rel_items["gold"], rel_items["pred"]
        for item in p_set:
            if item[1] == "Regulation":
                reg_p.add(item)
            elif item[1] == "Positive_regulation":
                pos_reg_p.add(item)
            elif item[1] == "Negative_regulation":
                neg_reg_p.add(item)
        for item in g_set:
            if item[1] == "Regulation":
                reg_g.add(item)
            elif item[1] == "Positive_regulation":
                pos_reg_g.add(item)
            elif item[1] == "Negative_regulation":
                neg_reg_g.add(item)

    corr_num = len(reg_p & reg_g)
    pred_num = len(reg_p)
    gold_num = len(reg_g)
    P = corr_num / (pred_num + 1e-10)
    R = corr_num / (gold_num + 1e-10)
    F = 2 * P * R / (P + R + 1e-10)
    logger.info(f"+" * 30)
    logger.info("precision (Relation-Reg): {0:>.4%}".format(P))
    logger.info("recall (Relation-Reg): {0:>.4%}".format(R))
    logger.info("f1 (Relation-Reg): {0:>.4%}".format(F))
    logger.info(f"+" * 30)

    corr_num = len(pos_reg_p & pos_reg_g)
    pred_num = len(pos_reg_p)
    gold_num = len(pos_reg_g)
    P = corr_num / (pred_num + 1e-10)
    R = corr_num / (gold_num + 1e-10)
    F = 2 * P * R / (P + R + 1e-10)
    logger.info(f"+" * 30)
    logger.info("precision (Relation-Pos_Reg): {0:>.4%}".format(P))
    logger.info("recall (Relation-Pos_Reg): {0:>.4%}".format(R))
    logger.info("f1 (Relation-Pos_Reg): {0:>.4%}".format(F))
    logger.info(f"+" * 30)

    corr_num = len(neg_reg_p & neg_reg_g)
    pred_num = len(neg_reg_p)
    gold_num = len(neg_reg_g)
    P = corr_num / (pred_num + 1e-10)
    R = corr_num / (gold_num + 1e-10)
    F = 2 * P * R / (P + R + 1e-10)
    logger.info(f"+" * 30)
    logger.info("precision (Relation-Neg_Reg): {0:>.4%}".format(P))
    logger.info("recall (Relation-Neg_Reg): {0:>.4%}".format(R))
    logger.info("f1 (Relation-Neg_Reg): {0:>.4%}".format(F))
    logger.info(f"+" * 30)

    lost_count, total_count = 0, 0
    pred, gold = set(), set()
    for doc_id, net_items in net_std.items():
        net_pred, net_gold = set(), set()
        for net_item in net_items["pred"]:
            total_count += 1
            _, _, role, h_start, t_start, _ = net_item[:6]
            head_part = None
            if doc_id in mix1:
                pred_mixes = mix1[doc_id]
                for pred_mix in pred_mixes:
                    _, pred_m_type, pred_m_start, pred_m_end, pred_m_text = pred_mix
                    if pred_m_start <= h_start < pred_m_end:
                        head_part = pred_mix
                        break
                else:
                    doc_txt = open(f"{source_path}/{doc_id}.txt", encoding="utf-8", mode="r").read()
                    h_txt = doc_txt[h_start:].split()[0]
                    h_end = h_start + len(h_txt)
                    if h_txt.lower() in prior:
                        head_part = (doc_id, prior[h_txt.lower()][0][0], h_start, h_end, h_txt)
                        build_type = prior[h_txt.lower()][0][0]
                        tri_std[doc_id]["pred"].add((doc_id, build_type, h_start, h_end, h_txt))

            tail_part = None
            if doc_id in mix1:
                pred_items = sorted(list(mix1[doc_id]))
                for pred_item in pred_items:
                    _, pred_m_type, pred_m_start, pred_m_end, pred_m_text = pred_item
                    if pred_m_start <= t_start < pred_m_end:
                        tail_part = pred_item
                        break
                else:
                    doc_txt = open(f"{source_path}/{doc_id}.txt", encoding="utf-8", mode="r").read()
                    t_txt = doc_txt[t_start:].split()[0]
                    t_end = t_start + len(t_txt)
                    if t_txt.lower() in prior:
                        tail_part = (doc_id, prior[t_txt.lower()][0][0], t_start, t_end, t_txt)
                        build_type = prior[t_txt.lower()][0][0]
                        tri_std[doc_id]["pred"].add((doc_id, build_type, t_start, t_end, t_txt))

            if head_part is None or tail_part is None:
                lost_count += 1
            else:
                net_pred.add((doc_id, head_part[1], head_part[2], head_part[3], head_part[4],
                              tail_part[1], tail_part[2], tail_part[3], tail_part[4], role))
                pred.add((doc_id, head_part[1], head_part[2], head_part[3], head_part[4],
                          tail_part[1], tail_part[2], tail_part[3], tail_part[4], role))
        net_std[doc_id]["pred"] = net_pred

        for net_item in net_items["gold"]:
            _, sid, role, h_start, t_start, _ = net_item[:6]
            head_part = None
            if doc_id in mix2:
                gold_mixes = mix2[doc_id]
                for gold_mix in gold_mixes:
                    _, gold_h_type, gold_h_start, gold_h_end, gold_h_text = gold_mix
                    if gold_h_start <= h_start < gold_h_end:
                        head_part = gold_mix
                        break
                else:
                    raise ("error")

            tail_part = None
            if doc_id in mix2:
                gold_items = sorted(list(mix2[doc_id]))
                for gold_item in gold_items:
                    _, gold_t_type, gold_t_start, gold_t_end, gold_t_text = gold_item
                    if gold_t_start <= t_start < gold_t_end:
                        tail_part = gold_item
                        break
                else:
                    raise ("error")
            assert doc_id in entities
            net_gold.add((doc_id, head_part[1], head_part[2], head_part[3], head_part[4],
                          tail_part[1], tail_part[2], tail_part[3], tail_part[4], role))
            gold.add((doc_id, head_part[1], head_part[2], head_part[3], head_part[4],
                      tail_part[1], tail_part[2], tail_part[3], tail_part[4], role))

        net_std[doc_id]["gold"] = net_gold
    logger.info(f"wreck rate: {round(lost_count / (total_count + 1e-10), 4)}({lost_count}|{total_count})")
    for doc_id, net_items in net_std.items():
        gold.update(net_items["gold"])
        pred.update(net_items["pred"])
        for net_item in net_items["gold"]:
            if doc_id not in network_g:
                network_g[doc_id] = []
            net_item_str = f"{rel_item[0]}&&{net_item[2]}-{net_item[3]}:{net_item[4]}>>{net_item[1]}->{net_item[-1]}<-{net_item[5]}<<{net_item[8]}:{net_item[6]}-{net_item[7]}"
            if net_item_str not in network_g[doc_id]:
                network_g[doc_id].append(net_item_str)
        for net_item in net_items["pred"]:
            if doc_id not in network_p:
                network_p[doc_id] = []
            net_item_str = f"{rel_item[0]}&&{net_item[2]}-{net_item[3]}:{net_item[4]}>>{net_item[1]}->{net_item[-1]}<-{net_item[5]}<<{net_item[8]}:{net_item[6]}-{net_item[7]}"
            if net_item_str not in network_p[doc_id]:
                network_p[doc_id].append(net_item_str)
    pt, rt, ft = performance_report(y_true=gold, y_pred=pred)
    logger.info(f"devel metric".center(30, "*"))
    logger.info("precision (Network): {0:>.4%}".format(pt))
    logger.info("recall (Network): {0:>.4%}".format(rt))
    logger.info("f1 (Network): {0:>.4%}".format(ft))
    logger.info(f"*" * 30)

    pred_split, gold_split = {}, {}
    for doc_ids, net_items in net_std.items():
        g_set, p_set = net_items["gold"], net_items["pred"]
        for g_item in g_set:
            if g_item[-1] not in gold_split:
                gold_split[g_item[-1]] = set()
            gold_split[g_item[-1]].add(g_item)
        for p_item in p_set:
            if p_item[-1] not in pred_split:
                pred_split[p_item[-1]] = set()
            pred_split[p_item[-1]].add(p_item)
    for key, p_set in pred_split.items():
        g_set = gold_split[key]
        corr_num = len(g_set & p_set)
        pred_num = len(p_set)
        gold_num = len(g_set)
        P = corr_num / (pred_num + 1e-10)
        R = corr_num / (gold_num + 1e-10)
        F = 2 * P * R / (P + R + 1e-10)
        logger.info(f"+" * 30)
        logger.info("precision (Network-{0}): {1:>.4%}".format(key, P))
        logger.info("recall (Network-{0}): {1:>.4%}".format(key, R))
        logger.info("f1 (Network-{0}): {1:>.4%}".format(key, F))
        logger.info(f"+" * 30)

    pred_split, gold_split = {}, {}
    pred, gold = set(), set()
    for doc_ids, tri_items in tri_std.items():
        g_set, p_set = tri_items["gold"], tri_items["pred"]
        trigger_g[doc_ids] = [f"{tri_g[0]}&&{tri_g[2]}-{tri_g[3]}:{tri_g[4]}>>{tri_g[1]}" for tri_g in list(g_set)]
        trigger_p[doc_ids] = [f"{tri_p[0]}&&{tri_p[2]}-{tri_p[3]}:{tri_p[4]}>>{tri_p[1]}" for tri_p in list(p_set)]
        pred.update(p_set)
        gold.update(g_set)
        for g_item in g_set:
            if g_item[1] not in gold_split:
                gold_split[g_item[1]] = set()
            gold_split[g_item[1]].add(g_item)
        for p_item in p_set:
            if p_item[1] not in pred_split:
                pred_split[p_item[1]] = set()
            pred_split[p_item[1]].add(p_item)
    for key, p_set in pred_split.items():
        g_set = gold_split[key]
        corr_num = len(g_set & p_set)
        pred_num = len(p_set)
        gold_num = len(g_set)
        P = corr_num / (pred_num + 1e-10)
        R = corr_num / (gold_num + 1e-10)
        F = 2 * P * R / (P + R + 1e-10)
        logger.info(f"+" * 30)
        logger.info("precision (Trigger-{0}): {1:>.4%}".format(key, P))
        logger.info("recall (Trigger-{0}): {1:>.4%}".format(key, R))
        logger.info("f1 (Trigger-{0}): {1:>.4%}".format(key, F))
        logger.info(f"+" * 30)

    pt, rt, ft = performance_report(y_true=gold, y_pred=pred)
    logger.info(f"devel metric".center(30, "*"))
    logger.info("precision (Trigger): {0:>.4%}".format(pt))
    logger.info("recall (Trigger): {0:>.4%}".format(rt))
    logger.info("f1 (Trigger): {0:>.4%}".format(ft))  # 78.2148%
    logger.info(f"*" * 30)

    for doc_id in trigger_g:
        if doc_id not in trigger_p:
            trigger_p[doc_id] = []
    for doc_id in relation_g:
        if doc_id not in relation_p:
            relation_p[doc_id] = []
    for doc_id in network_g:
        if doc_id not in network_p:
            network_p[doc_id] = []

    for key in entities:
        entities[key] = [f"{ent[0]}&&{ent[2]}-{ent[3]}:{ent[4]}>>{ent[1]}" for ent in entities[key]]

    return trigger_p, trigger_g, relation_p, relation_g, network_p, network_g, entities


# globalpointer: F: 62.5028%
def trigger_select(trigger_p, mode="devel"):
    updated_trigger_p = {}
    for target in all_events:
        for doc, items in trigger_p.items():
            if doc not in updated_trigger_p:
                updated_trigger_p[doc] = set()
            for item in items:
                if target in item and target == "Transcription":
                    txt = item.split(":")[1].split(">>")[0]
                    name = item.split("&&")[0]
                    if txt.lower() not in ["amount", "level", "levels", "mrna levels", "mrna level",
                                           "translation", "translational", "protein level",
                                           "protein levels"]:
                        if len(txt.split()) > 1 and txt.split()[0].lower() == "transcription":
                            continue
                        if txt.lower() in ["mrna induction", "mrna up"]:
                            text = "mRNA"
                            span = item.split("&&")[1].split(":")[0]
                            start, end = int(span.split("-")[0]), int(span.split("-")[1])
                            end = start + len(text)
                            new_span = f"{start}-{end}"
                            item = item.replace(txt, text).replace(span, new_span)
                        # p_set.add(item)
                        updated_trigger_p[doc].add(item)
                if target in item and target == "Gene_expression":
                    txt = item.split(":")[1].split(">>")[0]
                    name = item.split("&&")[0]
                    if txt.lower() not in ["found", "injection", "injected", "injects", "injecting",
                                           "source", "sources", "lacking", "lacks", "lacked", "negative",
                                           "positive", "-/-", "contain", "contains", "containing", "contained",
                                           "secretion", "secreted", "secreting", "secrets", "secret", "abundance",
                                           "present"]:
                        if txt.endswith("ly") or txt.endswith("tive"):
                            continue
                        # p_set.add(item)
                        updated_trigger_p[doc].add(item)
                if target in item and target == "Localization":
                    # p_set.add(item)
                    updated_trigger_p[doc].add(item)
                if target in item and target == "Phosphorylation":
                    # p_set.add(item)
                    updated_trigger_p[doc].add(item)
                if target in item and target == "Binding":
                    txt = item.split(":")[1].split(">>")[0]
                    if txt.lower() in ["uses", "using", "used", "use", "ligand", "affinity", "colocalized",
                                       "attraction",
                                       "inmmunoprecipitated"]:
                        continue
                    if len(txt.split()) > 1:
                        continue
                    if txt.endswith("mer") or txt.endswith("mers") or txt.endswith("lity"):
                        continue
                    if "link" in txt.lower():
                        continue
                    # p_set.add(item)
                    updated_trigger_p[doc].add(item)
                if target in item and target == "Regulation":
                    txt = item.split(":")[1].split(">>")[0]
                    if txt.lower() in ["loop", "different", "dependently", "contribute", "contributes", 'contributed',
                                       "contributing", "uses", "using", "used", "use", "relationship", "relationships",
                                       "requirement", "requirements", "enhance", "enhances", "enhancement", "enhanced",
                                       "competition", "compete", "competes", "competed", "competing", "mediate",
                                       "mediated", "mediating", "mediates", "limit", "limits", "limited", "limiting",
                                       "decease", "decreased", "decreasing", "unchange", "unchanged", "unchanges",
                                       "unchanging",
                                       "influence", "influenced", "influences", "influencing", "block", "blocks",
                                       "blocking",
                                       "blocked", "induce", "induced", "inducing", "induces"]:
                        continue
                    if " to " in txt or "phosphorylated" in txt or "phosphorylation" in txt:
                        continue
                    if txt.endswith("ble") or txt.endswith("mers") or txt.endswith("lity"):
                        continue
                    if txt.startswith("post") or txt.startswith("interfere"):
                        continue
                    if txt.startswith("effective") and len(txt.split()) > 1:
                        continue
                    name = item.split("&&")[0]
                    text = open(f"./sources/GE13/{mode}/{name}.txt", encoding="utf-8", mode="r").read()
                    txt = item.split(":")[1].split(">>")[0]
                    span = item.split("&&")[1].split(":")[0]
                    start, end = int(span.split("-")[0]), int(span.split("-")[1])
                    if "role" in txt and ("play a" in text[max(0, start - 30):start].strip() or "play an" in text[max(0,
                                                                                                                      start - 30):start].strip() or \
                                          "plays a" in text[max(0, start - 30):start].strip() or "plays an" in text[
                                                                                                               max(0,
                                                                                                                   start - 30):start].strip() or \
                                          "played a" in text[max(0, start - 30):start].strip() or "played an" in text[
                                                                                                                 max(0,
                                                                                                                     start - 30):start].strip()):
                        while True:
                            if text[start: start + 4] == "play":
                                break
                            else:
                                start -= 1
                        txt = text[start:end]
                        item = f"{name}&&{start}-{end}:{txt}>>Regulation"
                        # p_set.add(line)
                    if "role" in txt.lower() and "play" in text and text[end:].split()[0] in ["of", "for"]:
                        continue
                    if "change" in txt and len(txt.split()) > 1:
                        continue
                    temp_text = text[:start].split()[-3:]
                    # if "dependent" == txt and temp_text[-1].strip("-") in ["time", "dose", "concentration"]:
                    #     continue
                    if "effect" in txt and len(txt.split()) > 1 or txt == "act" or (
                            "under" in txt and len(txt.split()) > 2):
                        continue
                    if "Role" == txt or "crucial" in txt or "critical" in txt or "mutat" in txt.lower() or "major" in txt:
                        continue
                    # p_set.add(item)
                    updated_trigger_p[doc].add(item)
                if target in item and target == "Protein_catabolism":
                    name = item.split("&&")[0]
                    text = open(f"./sources/GE13/{mode}/{name}.txt", encoding="utf-8", mode="r").read()
                    txt = item.split(":")[1].split(">>")[0]
                    span = item.split("&&")[1].split(":")[0]
                    start, end = int(span.split("-")[0]), int(span.split("-")[1])
                    if txt in ["processing", "process", "processed", "processes"]:
                        continue
                    if txt.startswith("degradation") and txt[-1].isdigit():
                        continue
                    # p_set.add(item)
                    updated_trigger_p[doc].add(item)
                if target in item and target == "Positive_regulation":
                    txt = item.split(":")[1].split(">>")[0]
                    if txt.lower() in ["larger", "unstable", "response", "dominant", "raiped", "enabling", "enables",
                                       "enabled",
                                       "enable",
                                       "abundant", "sustain", "sustains", "sustained", "sustaining", "achieve",
                                       "achieved",
                                       "achieving",
                                       "achieves", "after", "before", "maximal"]:
                        continue

                    # p_set.add(item)
                    updated_trigger_p[doc].add(item)
                if target in item and target == "Negative_regulation":
                    txt = item.split(":")[1].split(">>")[0]
                    # if txt.lower() in ["larger", "unstable", "response", "dominant", "raiped", "enabling", "enables", "enabled", "enable",
                    #                    "abundant", "sustain", "sustains", "sustained", "sustaining", "achieve", "achieved", "achieving",
                    #                    "achieves", "after", "before", "maximal"]:
                    #     continue

                    # p_set.add(item)
                    updated_trigger_p[doc].add(item)
    return updated_trigger_p


def event_merge(std_data, event_dist, batch_size, tokenizers, char_maps, sent_starts, logger):
    instances = []
    for sent_idx, item_dict in tqdm(std_data.items(), desc="reconstructed data"):
        tri_spans = item_dict["trigger"]
        arguments = item_dict["argument"]
        tokens = item_dict["token"]
        theme_spans, cause_spans = [], []
        for argument in arguments:
            if argument[-1] == "Theme":
                theme_spans.append(argument)
            else:
                cause_spans.append(argument)
        constructed_events = []
        tri_spans = list(tri_spans)[:10]
        theme_spans = theme_spans[:10]
        cause_spans = cause_spans[:10]
        for tri_span in tri_spans:
            t_span_start, t_span_end, t_text, tri_type = tri_span
            if tri_type in Theme_events:  # only contain theme argument
                for theme_candidate in theme_spans:
                    a_start, a_end, a_text, a_type, role = theme_candidate
                    dist_pos = t_span_start - a_start
                    if not (event_dist[tri_type]["Theme"][0] <= dist_pos <= event_dist[tri_type]["Theme"][-1]):
                        continue
                    answer = [tri_span, [{"role": "Theme", "content": theme_candidate}]]
                    if answer not in constructed_events:
                        constructed_events.append(answer)
            elif tri_type in MThemes_events:
                for theme_candidate in theme_spans:
                    a_start, a_end, a_text, a_type, role = theme_candidate
                    dist_pos = t_span_start - a_start
                    if not (event_dist[tri_type]["Theme"][0] <= dist_pos <= event_dist[tri_type]["Theme"][-1]):
                        continue

                    answer = [tri_span, [{"role": "Theme", "content": theme_candidate}]]
                    if answer not in constructed_events:
                        constructed_events.append(answer)
                if len(theme_spans) > 1:
                    for i, theme1_candidate in enumerate(theme_spans):
                        a_start1, a_end1, a_text1, a_type1, _ = theme1_candidate
                        for j, theme2_candidate in enumerate(theme_spans):
                            a_start2, a_end2, a_text2, a_type2, _ = theme2_candidate
                            if i < j:

                                dist = [t_span_start - a_start1, t_span_start - a_start2]
                                min_pos, max_pos = min(dist), max(dist)
                                if not (event_dist[tri_type]["Theme"][0] <= min_pos <= max_pos <=
                                        event_dist[tri_type]["Theme"][-1]):
                                    continue

                                answer = [tri_span, [{"role": "Theme", "content": theme1_candidate},
                                                     {"role": "Theme", "content": theme2_candidate}]]
                                if answer not in constructed_events:
                                    constructed_events.append(answer)
                if len(theme_spans) > 2:
                    for i, theme1_candidate in enumerate(theme_spans):
                        a_start1, a_end1, a_text1, a_type1, _ = theme1_candidate
                        for j, theme2_candidate in enumerate(theme_spans):
                            a_start2, a_end2, a_text2, a_type2, _ = theme2_candidate
                            for k, theme3_candidate in enumerate(theme_spans):
                                a_start3, a_end3, a_text3, a_type3, _ = theme3_candidate
                                if i < j < k:

                                    dist = [t_span_start - a_start1,
                                            t_span_start - a_start2,
                                            t_span_start - a_start3]
                                    min_pos, max_pos = min(dist), max(dist)
                                    if not (event_dist[tri_type]["Theme"][0] <= min_pos <= max_pos <=
                                            event_dist[tri_type]["Theme"][-1]):
                                        continue

                                    answer = [tri_span, [{"role": "Theme", "content": theme1_candidate},
                                                         {"role": "Theme", "content": theme2_candidate},
                                                         {"role": "Theme", "content": theme3_candidate}]]
                                    if answer not in constructed_events:
                                        constructed_events.append(answer)
            elif tri_type in Theme_Cause_events:
                for theme_candidate in theme_spans:
                    a_start, a_end, a_text, a_type, role = theme_candidate

                    dist_pos = t_span_start - a_start
                    if not (event_dist[tri_type]["Theme"][0] <= dist_pos <= event_dist[tri_type]["Theme"][-1]):
                        continue

                    answer = [tri_span, [{"role": "Theme", "content": theme_candidate}]]
                    if answer not in constructed_events:
                        constructed_events.append(answer)

                for cause_candidate in cause_spans:
                    a_start, a_end, a_text, a_type, role = cause_candidate

                    dist_pos = t_span_start - a_start
                    if event_dist[tri_type]["Cause"] is None or not (
                            event_dist[tri_type]["Cause"][0] <= dist_pos <= event_dist[tri_type]["Cause"][-1]):
                        continue

                    answer = [tri_span, [{"role": "Cause", "content": cause_candidate}]]
                    if answer not in constructed_events:
                        constructed_events.append(answer)

                for cause_candidate in cause_spans:
                    for theme_candidate in theme_spans:
                        a_start1, a_end1, a_text1, a_type1, _ = theme_candidate
                        a_start2, a_end2, a_text2, a_type2, _ = cause_candidate

                        theme_dist = t_span_start - a_start1
                        cause_dist = t_span_start - a_start2

                        if event_dist[tri_type]["Cause"] is None or not (
                                event_dist[tri_type]["Theme"][0] <= theme_dist <= event_dist[tri_type]["Theme"][-1] and \
                                event_dist[tri_type]["Cause"][0] <= cause_dist <= event_dist[tri_type]["Cause"][-1]):
                            continue

                        answer = [tri_span, [{"role": "Cause", "content": cause_candidate},
                                             {"role": "Theme", "content": theme_candidate}]]
                        if answer not in constructed_events:
                            constructed_events.append(answer)

        for tri_span, arguments in constructed_events:
            tri_tpos, tri_epos, tri_text, tri_type = tri_span
            if len(arguments) == 1:
                if tri_type in Theme_events or tri_type in MThemes_events:
                    a_start, a_end, a_text, a_type, _ = arguments[0]["content"]
                    # 构造事件问题
                    event_description = f"Is the #{tri_type}# type flat event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} and " \
                                        f"a @Theme@ <{a_type}> argument ${a_text}$ at position {a_start}-{a_end} correct ?"
                    instances.append(
                        {"id": sent_idx, "trigger": tri_span, "arguments": arguments,
                         "question": tokens + event_description.split()})
                elif tri_type in Nest_events:
                    a_start, a_end, a_text, a_type, _ = arguments[0]["content"]
                    if arguments[0]["role"] == "Theme":
                        if a_type in all_events:
                            if a_type in all_events:
                                event_description = f"Is the #{tri_type}# type nested event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} and " \
                                                    f"a @Theme@ <{a_type}> event argument ${a_text}$ at position {a_start}-{a_end} correct ?"
                            else:
                                event_description = f"Is the #{tri_type}# type flat event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} and " \
                                                    f"a @Theme@ <{a_type}> argument ${a_text}$ at position {a_start}-{a_end} correct ?"
                    if arguments[0]["role"] == "Cause":
                        if a_type in all_events:
                            if a_type in all_events:
                                event_description = f"Is the #{tri_type}# type nested event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} and " \
                                                    f"a @Cause@ <{a_type}> event argument ${a_text}$ at position {a_start}-{a_end} correct ?"
                            else:
                                event_description = f"Is the #{tri_type}# type flat event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} and " \
                                                    f"a @Cause@ <{a_type}> argument ${a_text}$ at position {a_start}-{a_end} correct ?"
                    instances.append(
                        {"id": sent_idx, "trigger": tri_span, "arguments": arguments,
                         "question": tokens + event_description.split()})
                else:
                    a_start, a_end, a_text, a_type, _ = arguments[0]["content"]
                    if arguments[0]["role"] == "Theme":
                        event_description = f"Is the #{tri_type}# type flat event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} and " \
                                            f"a @Theme@ <{a_type}> argument ${a_text}$ at position {a_start}-{a_end} correct ?"
                    if arguments[0]["role"] == "Cause":
                        event_description = f"Is the #{tri_type}# type flat event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} and " \
                                            f"a @Cause@ <{a_type}> argument ${a_text}$ at position {a_start}-{a_end} correct ?"
                    instances.append(
                        {"id": sent_idx, "trigger": tri_span, "arguments": arguments,
                         "question": tokens + event_description.split()})

            if len(arguments) == 2:
                if tri_type in MThemes_events:
                    a_start1, a_end1, a_text1, a_type1, _ = arguments[0]["content"]
                    a_start2, a_end2, a_text2, a_type2, _ = arguments[1]["content"]

                    event_description = f"Is the #{tri_type}# type flat event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} , " \
                                        f"a @Theme@ <{a_type1}> event argument ${a_text1}$ at position {a_start1}-{a_end1} and " \
                                        f"a @Theme@ <{a_type2}> event argument ${a_text2}$ at position {a_start2}-{a_end2} correct ?"
                    instances.append(
                        {"id": sent_idx, "trigger": tri_span, "arguments": arguments,
                         "question": tokens + event_description.split()})
                elif tri_type in Nest_events:
                    a_start1, a_end1, a_text1, a_type1, _ = arguments[0]["content"]
                    a_start2, a_end2, a_text2, a_type2, _ = arguments[1]["content"]

                    if a_type1 in all_events and a_type2 in all_events:
                        event_description = f"Is the #{tri_type}# type nested event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} , " \
                                            f"a @Cause@ <{a_type1}> event argument ${a_text1}$ at position {a_start1}-{a_end1} and " \
                                            f"a @Theme@ <{a_type2}> event argument ${a_text2}$ at position {a_start2}-{a_end2} correct ?"
                    elif a_type1 not in all_events and a_type2 in all_events:
                        event_description = f"Is the #{tri_type}# type nested event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} , " \
                                            f"a @Cause@ <{a_type1}> argument ${a_text1}$ at position {a_start1}-{a_end1} and " \
                                            f"a @Theme@ <{a_type2}> event argument ${a_text2}$ at position {a_start2}-{a_end2} correct ?"
                    elif a_type1 in all_events and a_type2 not in all_events:
                        event_description = f"Is the #{tri_type}# type nested event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} , " \
                                            f"a @Cause@ <{a_type1}> event argument ${a_text1}$ at position {a_start1}-{a_end1} and " \
                                            f"a @Theme@ <{a_type2}> argument ${a_text2}$ at position {a_start2}-{a_end2} correct ?"
                    else:
                        event_description = f"Is the #{tri_type}# type flat event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} , " \
                                            f"a @Cause@ <{a_type1}> argument ${a_text1}$ at position {a_start1}-{a_end1} and " \
                                            f"a @Theme@ <{a_type2}> argument ${a_text2}$ at position {a_start2}-{a_end2} correct ?"
                    instances.append(
                        {"id": sent_idx, "trigger": tri_span, "arguments": arguments,
                         "question": tokens + event_description.split()})
                else:
                    a_start1, a_end1, a_text1, a_type1, _ = arguments[0]["content"]
                    a_start2, a_end2, a_text2, a_type2, _ = arguments[1]["content"]

                    event_description = f"Is the #{tri_type}# type flat event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} , " \
                                        f"a @Cause@ <{a_type1}> argument ${a_text1}$ at position {a_start1}-{a_end1} and " \
                                        f"a @Theme@ <{a_type2}> argument ${a_text2}$ at position {a_start2}-{a_end2} correct ?"
                    instances.append(
                        {"id": sent_idx, "trigger": tri_span, "arguments": arguments,
                         "question": tokens + event_description.split()})
            if len(arguments) == 3:
                a_start1, a_end1, a_text1, a_type1, _ = arguments[0]["content"]
                a_start2, a_end2, a_text2, a_type2, _ = arguments[1]["content"]
                a_start3, a_end3, a_text3, a_type3, _ = arguments[2]["content"]

                event_description = f"Is the #{tri_type}# type flat event consisting a trigger ${tri_text}$ at position {tri_tpos}-{tri_epos} , " \
                                    f"a @Theme@ <{a_type1}> event argument ${a_text1}$ at position {a_start1}-{a_end1} , " \
                                    f"a @Theme@ <{a_type2}> event argument ${a_text2}$ at position {a_start2}-{a_end2} and " \
                                    f"a @Theme@ <{a_type3}> event argument ${a_text3}$ at position {a_start3}-{a_end3} correct ?"
                instances.append(
                    {"id": sent_idx, "trigger": tri_span, "arguments": arguments,
                     "question": tokens + event_description.split()})

    logger.info(f"reconstructed instance number: {len(instances)}")
    logger.info("loading dataloader:")
    dataset = MyDataset(instances)
    dataloader = DataLoader(dataset, batch_size, shuffle=False, num_workers=0,
                            collate_fn=partial(my_collate, tokenizer=tokenizers, char_maps=char_maps,
                                               sent_starts=sent_starts),
                            pin_memory=True)
    return dataloader


def bio_tags_to_spans(tag_sequence, classes_to_ignore=[]):
    """
    Given a sequence corresponding to BIO tags, extracts spans.
    Spans are inclusive and can be of zero length, representing a single word span.
    Ill-formed spans are also included (i.e those which do not start with a "B-LABEL"),
    as otherwise it is possible to get a perfect precision score whilst still predicting
    ill-formed spans in addition to the correct spans. This function works properly when
    the spans are unlabeled (i.e., your labels are simply "B", "I", and "O").
    # Parameters
    tag_sequence : `List[str]`, required.
        The integer class labels for a sequence.
    classes_to_ignore : `List[str]`, optional (default = `None`).
        A list of string class labels `excluding` the bio tag
        which should be ignored when extracting spans.
    # Returns
    spans : `List[TypedStringSpan]`
        The typed, extracted spans from the sequence, in the format (label, (span_start, span_end)).
        Note that the label `does not` contain any BIO tag prefixes.
    """
    classes_to_ignore = classes_to_ignore or []
    spans = set()
    span_start = 0
    span_end = 0
    active_conll_tag = None
    for index, string_tag in enumerate(tag_sequence):
        # Actual BIO tag.
        bio_tag = string_tag[0]
        if bio_tag not in ["B", "I", "O"]:
            raise InvalidTagSequence(tag_sequence)
        conll_tag = string_tag[2:]
        if bio_tag == "O" or conll_tag in classes_to_ignore:
            # The span has ended.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = None
            # We don't care about tags we are
            # told to ignore, so we do nothing.
            continue
        elif bio_tag == "B":
            # We are entering a new span; reset indices
            # and active tag to new span.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
        elif bio_tag == "I" and conll_tag == active_conll_tag:
            # We're inside a span.
            span_end += 1
        else:
            # This is the case the bio label is an "I", but either:
            # 1) the span hasn't started - i.e. an ill formed span.
            # 2) The span is an I tag for a different conll annotation.
            # We'll process the previous span if it exists, but also
            # include this span. This is important, because otherwise,
            # a model may get a perfect F1 score whilst still including
            # false positive ill-formed spans.
            if active_conll_tag is not None:
                spans.add((active_conll_tag, (span_start, span_end)))
            active_conll_tag = conll_tag
            span_start = index
            span_end = index
    # Last token might have been a part of a valid span.
    if active_conll_tag is not None:
        spans.add((active_conll_tag, (span_start, span_end)))
    return list(spans)


import torch
from functools import partial
from torch.utils.data import Dataset, DataLoader


class MyDataset(Dataset):

    def __init__(self, raw_data):
        self._dataset = raw_data

    def __getitem__(self, index):
        return self._dataset[index]

    def __len__(self):
        return len(self._dataset)


def my_collate(batch, tokenizer, char_maps, sent_starts):
    answers = []
    inputs = []
    for instance in batch:
        sent_idx = instance["id"]
        tri_span = list(instance["trigger"])
        arg_spans = instance["arguments"]
        question = instance["question"]
        input_ids = tokenizer.encode(" ".join(question))
        tri_span[0] = char_maps[sent_idx][tri_span[0]][0] + sent_starts[sent_idx]
        tri_span[1] = char_maps[sent_idx][tri_span[1] - 1][1] + sent_starts[sent_idx]
        inputs.append(input_ids)
        new_args = []
        for item in arg_spans:
            item[0] = char_maps[sent_idx][item[0]][0] + sent_starts[sent_idx]
            item[1] = char_maps[sent_idx][item[1] - 1][1] + sent_starts[sent_idx]
            new_args.append(item)
        answers.append([sent_idx, tri_span, new_args])
    max_len = max([len(item) for item in inputs])
    new_inputs = []
    for item in inputs:
        pad_ids = [0] * (max_len - len(item))
        new_inputs.append(item + pad_ids)
    input_ids = torch.LongTensor(new_inputs)
    return input_ids, answers
