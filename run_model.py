import json
import os
import pickle
import random
import torch
import time
import argparse
import datetime
import numpy as np
import torch.nn as nn
from logger import Log
from tqdm import tqdm
from collections import Counter
from transformers import AutoTokenizer
from dataloader import SpanDataloader, MyDataset, my_collate
from torch.utils.data import DataLoader
from model_base import MentionPointer
from functools import partial
from metrics import performance_report, span_generation, event_merge, detail_scores
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW
from sklearn.metrics import fbeta_score
from merge import merge

torch.cuda.current_device()
torch.cuda._initialized = True

import warnings

warnings.filterwarnings("ignore")

Theme_events = ["Gene_expression", "Transcription", "Localization", "Protein_catabolism"]
Theme_Cause_events = ["Regulation", "Negative_regulation", "Positive_regulation", "Phosphorylation", "Ubiquitination",
                      "Deacetylation", "Protein_modification", "Acetylation"]
MThemes_events = ["Binding"]
Nest_events = ["Regulation", "Positive_regulation", "Negative_regulation"]
all_events = Theme_events + Theme_Cause_events + MThemes_events
all_roles = ["Theme", "Theme2", "Cause"]
all_edges = ["Theme$Cause", "Theme$Theme2"]


class PipeLineTrainer:
    def __init__(self, args, logger):
        self.args = args
        self.seed = args.seed
        self.logger = logger
        os.environ['PYTHONHASHSEED'] = str(args.seed)
        os.environ['CUDA_VISIBLE_DEVICES'] = str(args.seed)
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.cuda.set_device(args.gpu_id)

        self.logger.info("System information".center(60, '+'))
        self.logger.info(f'cuda available: {torch.cuda.is_available()}')

        self.logger.info(f"using cuda device: {torch.cuda.current_device()}")
        self.logger.info(f'torch version: {torch.__version__}')
        self.logger.info(f'torch cuda version: {torch.version.cuda}')
        time_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))
        self.logger.info(f'start running time: {time_str}')
        self.logger.info("Data preprocessing".center(60, "+"))

        self.output_path = args.output_path
        self.train_path = args.train_path
        self.devel_path = args.devel_path
        self.test_path = args.test_path

        os.makedirs(self.output_path, exist_ok=True)

        self.gradient_accumulation_steps = args.gradient_accumulation_steps
        self.train_batch_size = args.train_batch_size
        self.devel_batch_size = args.devel_batch_size
        self.test_batch_size = args.test_batch_size

        self.bert_path = args.bert_path
        self.max_grad_norm = args.max_grad_norm
        self.do_train = args.do_train
        self.do_devel = args.do_devel
        self.do_test = args.do_test

        self.weight_decay = self.args.weight_decay
        self.adam_epsilon = self.args.adam_epsilon
        self.warmup_proportion = self.args.warmup_proportion
        self.epoch = self.args.epoch
        self.bert_name = self.bert_path.split("/")[-1] if "/" in self.bert_path else self.bert_path.split("\\")[-1]

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if self.gradient_accumulation_steps < 1:
            raise ValueError(
                f"Invalid gradient_accumulation_steps parameter: {self.gradient_accumulation_steps}, must be >= 1")

        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.bert_path)
        self.processor = SpanDataloader(train_path=self.train_path,
                                        devel_path=self.devel_path,
                                        test_path=self.test_path,
                                        tokenizer=self.tokenizer,
                                        max_len=self.args.max_len,
                                        logger=self.logger,
                                        task_name=self.args.task_name)
        schema = self.processor.build_map()
        self.lab2id = schema["lab"]
        self.id2lab = {v: k for k, v in self.lab2id.items()}

        self.rel2id = schema["rel"]
        self.id2rel = {v: k for k, v in self.rel2id.items()}
        self.prior = json.load(open(self.args.prior_path, encoding="utf-8", mode="r"))
        self.logger.info(f"***** Using {torch.cuda.device_count()} GPU(s)! *****")

        self.devel_span = json.load(open(f"sources/{self.args.task_name}/span-devel.json", encoding="utf-8", mode="r"))
        self.test_span = json.load(open(f"sources/{self.args.task_name}/span-test.json", encoding="utf-8", mode="r"))

        if self.do_train:
            self.logger.info(f"data mode [sup train], starting:")
            data_file = os.path.join(self.output_path, f"train-{self.bert_name}.pt")
            if os.path.exists(data_file):
                self.logger.info(f"Loading data from {data_file}")
                self.train_data_len, train_tensor_data, info = pickle.load(open(data_file, mode="rb"))
                for item in info:
                    self.logger.info(item)
            else:
                train_tensor_data, info = self.processor.get_tensor(data_path=self.train_path,
                                                                    dataset_name="train",
                                                                    schema=schema)

                for item in info:
                    self.logger.info(item)
                self.train_data_len = len(train_tensor_data)
                self.logger.info(f"instance number: {self.train_data_len}")
                info.append(f"instance number: {self.train_data_len}")
                pickle.dump([self.train_data_len, train_tensor_data, info], open(data_file, mode="wb"))

            self.train_data = DataLoader(dataset=MyDataset(train_tensor_data), batch_size=self.train_batch_size,
                                         shuffle=True, num_workers=0, pin_memory=True,
                                         collate_fn=partial(my_collate, lab_len=len(self.rel2id)))
            self.train_batch_len = len(self.train_data)
            self.logger.info(f"batch number: {self.train_batch_len}")

        if self.do_devel:
            self.logger.info(f"data mode [sup devel], starting:")
            data_file = os.path.join(self.output_path, f"devel-{self.bert_name}.pt")
            if os.path.exists(data_file):
                self.logger.info(f"Loading data from {data_file}")
                self.devel_data_len, devel_tensor_data, info = pickle.load(open(data_file, mode="rb"))
                for item in info:
                    self.logger.info(item)
            else:
                devel_tensor_data, info = self.processor.get_tensor(data_path=self.devel_path,
                                                                    dataset_name="devel",
                                                                    schema=schema)

                for item in info:
                    self.logger.info(item)
                self.devel_data_len = len(devel_tensor_data)
                self.logger.info(f"instance number: {self.devel_data_len}")
                info.append(f"instance number: {self.devel_data_len}")
                pickle.dump([self.devel_data_len, devel_tensor_data, info], open(data_file, mode="wb"))

            self.devel_data = DataLoader(dataset=MyDataset(devel_tensor_data), batch_size=self.devel_batch_size,
                                         shuffle=True, num_workers=0, pin_memory=True,
                                         collate_fn=partial(my_collate, lab_len=len(self.rel2id)))

            self.devel_batch_len = len(self.devel_data)
            self.logger.info(f"batch number: {self.devel_batch_len}")

        if self.do_test:
            self.logger.info(f"data mode [sup test], starting:")
            data_file = os.path.join(self.output_path, f"test-{self.bert_name}.pt")
            if os.path.exists(data_file):
                self.logger.info(f"Loading data from {data_file}")
                self.test_data_len, test_tensor_data, info = pickle.load(open(data_file, mode="rb"))
                for item in info:
                    self.logger.info(item)
            else:
                test_tensor_data, info = self.processor.get_tensor(data_path=self.test_path,
                                                                   dataset_name="test",
                                                                   schema=schema)

                for item in info:
                    self.logger.info(item)
                self.test_data_len = len(test_tensor_data)
                self.logger.info(f"instance number: {self.test_data_len}")
                info.append(f"instance number: {self.test_data_len}")
                pickle.dump([self.test_data_len, test_tensor_data, info], open(data_file, mode="wb"))

            self.test_data = DataLoader(dataset=MyDataset(test_tensor_data), batch_size=self.test_batch_size,
                                        shuffle=True, num_workers=0, pin_memory=True,
                                        collate_fn=partial(my_collate, lab_len=len(self.rel2id)))

            self.test_batch_len = len(self.test_data)
            self.logger.info(f"batch number: {self.test_batch_len}")

        lab_classes = len(self.rel2id)
        self.model = MentionPointer(args=self.args, ent_type_size=lab_classes,
                                    inner_dim=self.args.inner_dim, device=self.device, RoPE=self.args.RoPE)

    def prepare_train(self, data_len, batch_size, epochs):
        model = self.model.to(self.device)
        num_train_steps = int(data_len / batch_size / self.args.gradient_accumulation_steps) * epochs
        warmup_steps = int(self.args.warmup_proportion * num_train_steps)
        optimizer = AdamW(model.parameters(), eps=1e-8, lr=self.args.lr)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_train_steps)
        model.train()
        return model, optimizer, scheduler

    def run_train(self):
        model, optimizer, scheduler = self.prepare_train(data_len=self.train_data_len, epochs=self.epoch,
                                                         batch_size=self.train_batch_size)
        loss_cr_fun = nn.CrossEntropyLoss()

        best_epoch, best_f1 = 0.0, 0.0

        for epoch in range(self.epoch):
            torch.cuda.empty_cache()
            model.train()
            self.logger.info(f"staring training epoch: {epoch + 1}")
            all_loss, all_ent_loss, all_arg_loss, step = 0.0, 0.0, 0.0, 1
            use_obj = tqdm(self.train_data)
            for step, batch in enumerate(use_obj, start=1):
                # continue
                info, input_ids, offsets, seq_lens, ent_labs, labs = batch
                input_ids = input_ids.to(self.device)
                ent_labs = ent_labs.to(self.device)
                labs = labs.to(self.device)
                seq_lens = seq_lens.to(self.device)
                if type(offsets) != list:
                    offsets = offsets.to(self.device)

                inputs = {'input_ids': input_ids, 'offsets': offsets, "ent_labs": ent_labs, "seq_lens": seq_lens}

                logits = model(**inputs)

                batch_loss = model.loss_fun(logits, labs)

                optimizer.zero_grad()
                batch_loss.backward()

                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                all_loss += batch_loss.item()

                data_time = str(datetime.datetime.now())[:-3]
                use_obj.set_description(
                    "{0} - [\tINFO] - INFO - loss: {1:.4f}".format(
                        data_time,
                        all_loss / step))

            self.logger.info("overall-loss: {0:.4f}".format(all_loss / step))

            pt, rt, ft, aux = self.prediction(batch_data=self.devel_data, mode="devel")
            f1_event = merge(aux=aux, save_path="experiments/GE13", source_path="sources/GE13", mode="devel",
                             logger=self.logger)

            trigger_p, trigger_g, relation_p, relation_g, network_p, network_g, entity_g = aux
            self.logger.info(f"devel metric".center(30, "*"))
            self.logger.info("precision (Overall): {0:>.4%}".format(pt))
            self.logger.info("recall (Overall): {0:>.4%}".format(rt))
            self.logger.info("f1 (Overall): {0:>.4%}".format(ft))
            self.logger.info("f1 (Event): {0:>.4%}".format(f1_event))
            self.logger.info(f"*" * 30)

            if best_f1 < f1_event:
                best_f1 = f1_event
                best_epoch = epoch + 1
                torch.save(self.model.state_dict(), f"experiments/GE13/best_model_{self.seed}.bin")

                file_names = {f"devel_trigger_p_{self.seed}.json": trigger_p,
                              f"devel_trigger_g_{self.seed}.json": trigger_g,
                              f"devel_relation_p_{self.seed}.json": relation_p,
                              f"devel_relation_g_{self.seed}.json": relation_g,
                              f"devel_network_p_{self.seed}.json": network_p,
                              f"devel_network_g_{self.seed}.json": network_g,
                              f"devel_entity_g_{self.seed}.json": entity_g}
                for key, value in file_names.items():
                    with open(f"experiments/{self.args.task_name}/{key}", encoding="utf-8", mode="w") as f:
                        for doc_id in value.keys():
                            value[doc_id] = list(value[doc_id])
                        json.dump(value, f, indent=4, ensure_ascii=False)

            self.logger.info("current best f1: tri: {0:>.4%}".format(best_f1))

            self.logger.info(f"current best epoch: {best_epoch}")

            if abs(best_epoch - epoch) > self.args.early_stop:
                self.logger.info("early stop!")
                break

    def prediction(self, batch_data, mode, source_path="./sources/GE13/devel"):
        if mode == "test":
            self.model.load_state_dict(torch.load(f"./experiments/{self.args.task_name}/best_model_{self.seed}.bin"))
        self.model.to(self.device)
        self.model.eval()

        torch.cuda.empty_cache()
        use_obj = tqdm(batch_data)
        mention_p, mention_g = set(), set()
        entities, tri_std, net_std, rel_std = {}, {}, {}, {}
        for step, batch in enumerate(use_obj, start=1):
            # continue
            info, input_ids, offsets, seq_lens, ent_labs, labs = batch
            input_ids = input_ids.to(self.device)
            ent_labs = ent_labs.to(self.device)
            labs = labs.to(self.device)
            seq_lens = seq_lens.to(self.device)
            if type(offsets) != list:
                offsets = offsets.to(self.device)

            with torch.no_grad():
                inputs = {'input_ids': input_ids, 'offsets': offsets, "ent_labs": ent_labs, "seq_lens": seq_lens}

                logits = self.model(**inputs)

            pred, true, pred_set, true_set = span_generation(logits=logits, gold=labs, id2lab=self.id2rel,
                                                             info=info, inner_dict=[tri_std, rel_std, net_std,
                                                                                    entities],
                                                             logger=self.logger)
            mention_p.update(set(pred_set))
            mention_g.update(set(true_set))

        # 清理触发词
        doc_spans = None
        if mode == "devel":
            doc_spans = self.devel_span
        elif mode == "test":
            doc_spans = self.test_span
        else:
            raise ("error doc spans!")
        assert "PMC-2222968-21-Materials_and_Methods" in entities
        aux = [doc_spans, tri_std, rel_std, net_std, entities, self.prior, self.logger, source_path]
        trigger_p, trigger_g, relation_p, relation_g, network_p, network_g, entities = detail_scores(aux)
        pt, rt, ft = performance_report(y_true=mention_g, y_pred=mention_p)
        if mode == "test":
            file_names = {f"{mode}_trigger_p_{self.seed}.json": trigger_p,
                          f"{mode}_trigger_g_{self.seed}.json": trigger_g,
                          f"{mode}_relation_p_{self.seed}.json": relation_p,
                          f"{mode}_relation_g_{self.seed}.json": relation_g,
                          f"{mode}_network_p_{self.seed}.json": network_p,
                          f"{mode}_network_g_{self.seed}.json": network_g,
                          f"{mode}_entity_g_{self.seed}.json": entities}
            for key, value in file_names.items():
                with open(f"experiments/{self.args.task_name}/{key}", encoding="utf-8", mode="w") as f:
                    for doc_id in value.keys():
                        value[doc_id] = list(value[doc_id])
                    json.dump(value, f, indent=4, ensure_ascii=False)
            source_path = "sources/GE13"
            save_path = "experiments/GE13"
            aux = [trigger_p, trigger_g, relation_p, relation_g, network_p, network_g, entities]
            merge(aux=aux, save_path=save_path, source_path=source_path, mode=mode)
        return pt, rt, ft, [trigger_p, trigger_g, relation_p, relation_g, network_p, network_g, entities]


if __name__ == '__main__':
    task_names = ["MLEE", "GE13"]
    task_name = "GE13"
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task_name', type=str, default=task_name,
                        choices=task_names)
    parser.add_argument('--train_path', type=str, default=f"./sources/{task_name}/train.pt")
    parser.add_argument('--devel_path', type=str, default=f"./sources/{task_name}/devel.pt")
    parser.add_argument('--test_path', type=str, default=f"./sources/{task_name}/test.pt")
    parser.add_argument('--prior_path', type=str, default=f"./sources/{task_name}/prior.json")

    parser.add_argument('--output_path', type=str, default=f"./experiments/{task_name}")
    parser.add_argument('--bert_path', type=str, default="./bert/scibert_scivocab_uncased")  # 本地不能用

    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--ent_dim', type=int, default=50)
    parser.add_argument('--inner_dim', type=int, default=64)
    parser.add_argument('--RoPE', type=bool, default=True)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_proportion', type=float, default=0.0)
    parser.add_argument('--drop_rate', type=float, default=0.5)
    parser.add_argument('--max_len', type=int, default=512)

    parser.add_argument('--stage', type=str, default="trigger", choices=["trigger", "argument"])

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--devel_batch_size', type=int, default=32)
    parser.add_argument('--test_batch_size', type=int, default=32)

    parser.add_argument('--fix_bert_embedding', type=bool, default=False)

    parser.add_argument('--do_train', type=bool, default=True)
    parser.add_argument('--do_devel', type=bool, default=True)
    parser.add_argument('--do_test', type=bool, default=True)

    args = parser.parse_args()

    os.makedirs('./log', exist_ok=True)
    log = Log(f'./log/{args.task_name}' + ".log")
    exp_logger = log.getLog()

    params = [(key, value) for key, value in args.__dict__.items() if not key.startswith('__')]
    exp_logger.info("Hyper-parameter information".center(60, '+'))
    for key, value in params:
        exp_logger.info(f'{key}: {value}')

    Pipeline = PipeLineTrainer(args=args, logger=exp_logger)
    Pipeline.run_train()
    # Pipeline.prediction(Pipeline.test_data, mode="test", source_path="./sources/GE13/test")
