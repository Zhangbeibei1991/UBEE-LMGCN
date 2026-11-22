import torch

def performance_report(y_true, y_pred):
    pred_num = len(y_pred)
    gold_num = len(y_true)
    corr_num = len(y_true & y_pred)
    P = corr_num / (pred_num + 1e-10)
    R = corr_num / (gold_num + 1e-10)
    F = 2 * P * R / (P + R + 1e-10)
    return P, R, F


def generation(output, info, id2tri=None, id2rel=None, tag2id=None, prior=None, struct=None, task_name="GE13"):
    batch_g_triggers = set()
    batch_p_triggers = set()
    batch_p_relations = set()
    batch_g_relations = set()
    batch_stp_triggers = set()
    batch_stp_relations = set()
    t_results = torch.argmax(output["tri"], dim=-1).tolist()
    r_results = torch.argmax(output["rel"], dim=-1).tolist()
    s_results = torch.argmax(output["lab"]["predicts"], dim=-1).tolist()
    for trigger_list in info["tg"]:
        # batch_g_triggers.update(set([item for item in trigger_list if item.split(">>")[-1] in all_tri_type]))
        batch_g_triggers.update(set(trigger_list))
    for relation_list, sent_idx in zip(info["rg"], info["sent_idx"]):
        batch_g_relations.update(relation_list)
    mix_dict = {}
    for i, (tri_index, doc_idx, sent_idx, seq_len, char_map, org_txt, sentText, sent_start, tag) in enumerate(
            zip(t_results, info["doc_idx"], info["sent_idx"], info["seq_len"], info["char_map"],
                info["org_txt"], info["sentText"], info["sent_start"], info["tag"])):
        tri_lab = [id2tri[item] for item in tri_index][:seq_len]
        tri_results = bio_tags_to_spans(tag_sequence=tri_lab)
        mix_dict[sent_idx] = []
        for (type_, (span_start, span_end)) in tri_results:
            char_start = int(char_map[span_start][0])
            if task_name in ["GE13", "GE11"]:
                char_end = int(char_map[span_start][-1])
            else:
                char_end = int(char_map[span_end][-1])
            org_text = org_txt[char_start:char_end]
            trigger_text = ' '.join(sentText[span_start:span_end + 1])
            word = org_text.lower()
            if tag[span_start] not in tag2id:
                continue
            if task_name == "MLEE":
                if word.lower() not in ["due", "pathway", "off", "gene", "null", "mrna", "a", "an", "down", "insufficient",
                                        "insufficiently", "prime", "randomized", "microvessels", "followed",
                                        "immunohistochemical",
                                        "chemotactic", "acts", "achieved", "expanded", "subclassification", "organization",
                                        "cleared", "loading", "sufficient", "sufficiently", "recultivation", "vascularised",
                                        "propagated", "recognition", "recognition", "created", "negative", "bloated",
                                        "transfer",
                                        "connected", "stimulatory", "suppressive", "fix", "fixed", "fixes", "fixing",
                                        "circulating",
                                        "created", "necrotic", "supplying", "acceleration", "vasculogenic", "roles",
                                        "explants",
                                        "migratory", "pathogenesis", "intratumoral", "beneficial", "concentrate",
                                        "covalently",
                                        "protein", "co", "favorable", "negatively", "effective", "adhesive", "reparative",
                                        "cell",
                                        "levels", "critical", "important", "origin", "ectopic", "dna", "prepared",
                                        "intravitreal",
                                        "neo", "requirement", "viable", "leads", "cross", "turn", "dominant",
                                        "subcutaneous",
                                        "related"] and not word.startswith(
                    "-") and not word.endswith("-") and not word.startswith("an ") and not word.startswith(
                    "adjunct"):  # and (char_start >= 1 and not org_txt[char_start - 1] == "-"):

                    if word in prior:
                        types = [item[0] for item in prior[word]]
                        if type_ in types:
                            batch_p_triggers.add(
                                f"{doc_idx}&&{char_start + sent_start}-{char_end + sent_start}:{org_text}>>{type_}")
                            batch_stp_triggers.add(
                                f"{sent_idx}&&{span_start}-{span_end + 1}:{trigger_text}>>{type_}")
                            temp = (char_start + sent_start, char_end + sent_start, type_)
                            if temp not in mix_dict[sent_idx]:
                                mix_dict[sent_idx].append(temp)
                    else:
                        batch_p_triggers.add(
                            f"{doc_idx}&&{char_start + sent_start}-{char_end + sent_start}:{org_text}>>{type_}")
                        batch_stp_triggers.add(
                            f"{sent_idx}&&{span_start}-{span_end + 1}:{trigger_text}>>{type_}")
                        temp = (char_start + sent_start, char_end + sent_start, type_)
                        if temp not in mix_dict[sent_idx]:
                            mix_dict[sent_idx].append(temp)
            else:
                if word in prior:
                    types = [item[0] for item in prior[word]]
                    if type_ in types:
                        batch_p_triggers.add(
                            f"{doc_idx}&&{char_start + sent_start}-{char_end + sent_start}:{org_text}>>{type_}")
                        batch_stp_triggers.add(
                            f"{sent_idx}&&{span_start}-{span_end + 1}:{trigger_text}>>{type_}")
                        temp = (char_start + sent_start, char_end + sent_start, type_)
                        if temp not in mix_dict[sent_idx]:
                            mix_dict[sent_idx].append(temp)
                else:
                    batch_p_triggers.add(
                        f"{doc_idx}&&{char_start + sent_start}-{char_end + sent_start}:{org_text}>>{type_}")
                    batch_stp_triggers.add(
                        f"{sent_idx}&&{span_start}-{span_end + 1}:{trigger_text}>>{type_}")
                    temp = (char_start + sent_start, char_end + sent_start, type_)
                    if temp not in mix_dict[sent_idx]:
                        mix_dict[sent_idx].append(temp)

    for i, (rel_index, s_index, t_index, doc_idx, sent_idx, seq_len, char_map, org_txt, sentText, sent_start, pos,
            ents, tag) in enumerate(
        zip(r_results, s_results, t_results, info["doc_idx"], info["sent_idx"], info["seq_len"], info["char_map"],
            info["org_txt"], info["sentText"], info["sent_start"], info["pos"], info["ent"], info["tag"])):
        if s_index == 0:
            continue
        tri_lab = [id2tri[item] for item in t_index][:seq_len]
        t_results = bio_tags_to_spans(tag_sequence=tri_lab)
        mix_str = []
        for ent in ents:
            charoffset, ent_type = ent.split("&&")[1].split(":")[0], ent.split("&&")[1].split(">>")[-1]
            ent_start, ent_end = int(charoffset.split("-")[0]), int(charoffset.split("-")[1])
            temp = (ent_start, ent_end, ent_type)
            if temp not in mix_str:
                mix_str.append(temp)

        mix_str += mix_dict[sent_idx]

        tri_part = None
        for (type_, (span_start, span_end)) in t_results:
            char_start = int(char_map[span_start][0])
            if task_name in ["GE11", "GE13"]:
                char_end = int(char_map[span_start][-1])
            else:
                char_end = int(char_map[span_end][-1])
            org_text = org_txt[char_start:char_end]
            temp = f"{doc_idx}&&{char_start + sent_start}-{char_end + sent_start}:{org_text}>>{type_}"
            if pos == span_start and temp in batch_p_triggers:
                tri_part = f"{doc_idx}&&{char_start + sent_start}-{char_end + sent_start}:{org_text}>>{type_}"
                break

        if tri_part is not None:
            rel_lab = [id2rel[item] for item in rel_index][:seq_len]
            rel_results = bio_tags_to_spans(tag_sequence=rel_lab)
            for (role, (span_start, span_end)) in rel_results:
                char_start = int(char_map[span_start][0])
                if task_name in ["GE11", "GE13"]:
                    char_end = int(char_map[span_start][-1])
                else:
                    char_end = int(char_map[span_end][-1])
                org_text = org_txt[char_start:char_end]
                type_ = None
                for (start, end, mix_type) in mix_str: # mix 随机的有重复的
                    if start <= char_start + sent_start < char_end + sent_start <= end:
                        type_ = mix_type
                        char_start = start - sent_start
                        char_end = end - sent_start
                        break
                if type_ is not None:
                    t_type = tri_part.split(">>")[-1]
                    a_type = type_
                    if role in struct and (t_type, a_type) in struct[role]:
                        batch_p_relations.add(
                            tri_part + f"->{role}<-{type_}<<{char_start + sent_start}-{char_end + sent_start}:{org_text}")
    return batch_p_triggers, batch_g_triggers, batch_stp_triggers, batch_p_relations, batch_g_relations, batch_stp_relations


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
