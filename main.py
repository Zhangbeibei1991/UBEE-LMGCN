import json
import os
import random
import torch
import time
import argparse
import datetime
import numpy as np
import torch.nn as nn
from logger import Log
from tqdm import tqdm
from transformers import AutoTokenizer
from dataloader.dataloader import Dataloader
from model.model_base import DMGModel
from metrics.metrics import performance_report, generation
from torch.nn.utils import clip_grad_norm_
from transformers import get_linear_schedule_with_warmup
from torch.optim import AdamW

torch.cuda.current_device()
torch.cuda._initialized = True

import warnings

warnings.filterwarnings("ignore")


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

        tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=self.bert_path)
        self.processor = Dataloader(train_path=self.train_path,
                                    devel_path=self.devel_path,
                                    test_path=self.test_path,
                                    tokenizer=tokenizer,
                                    logger=self.logger,
                                    task_name=self.args.task_name)
        schema = self.processor.build_map()
        self.lab2id = schema["lab"]
        self.id2lab = {v: k for k, v in self.lab2id.items()}

        self.rel2id = schema["rel"]
        self.id2rel = {v: k for k, v in self.rel2id.items()}

        self.struct = schema["struct"]

        self.prior = json.load(open(self.args.prior_path, encoding="utf-8", mode="r"))
        self.tag2id = json.load(open(self.args.tag_path, encoding="utf-8", mode="r"))

        self.logger.info(f"***** Using {torch.cuda.device_count()} GPU(s)! *****")

        if self.do_train:
            self.logger.info(f"data mode [sup train], starting:")
            data_file = os.path.join(self.output_path, f"train-{self.bert_name}.pt")
            if os.path.exists(data_file):
                self.logger.info(f"Loading data from {data_file}")
                self.train_data_len, train_tensor_data, info = torch.load(data_file)
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
                torch.save([self.train_data_len, train_tensor_data, info], data_file)

            self.train_data = self.processor.get_batch_data(data=train_tensor_data,
                                                            mode="train",
                                                            batch_size=self.train_batch_size)

            self.train_batch_len = len(self.train_data)
            self.logger.info(f"batch number: {self.train_batch_len}")

        if self.do_devel:
            self.logger.info(f"data mode [sup devel], starting:")
            data_file = os.path.join(self.output_path, f"devel-{self.bert_name}.pt")
            if os.path.exists(data_file):
                self.logger.info(f"Loading data from {data_file}")
                self.devel_data_len, devel_tensor_data, info = torch.load(data_file)
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
                torch.save([self.devel_data_len, devel_tensor_data, info], data_file)

            self.devel_data = self.processor.get_batch_data(data=devel_tensor_data,
                                                            mode="devel",
                                                            batch_size=self.devel_batch_size)

            self.devel_batch_len = len(self.devel_data)
            self.logger.info(f"batch number: {self.devel_batch_len}")

        if self.do_test:
            self.logger.info(f"data mode [sup test], starting:")
            data_file = os.path.join(self.output_path, f"test-{self.bert_name}.pt")
            if os.path.exists(data_file):
                self.logger.info(f"Loading data from {data_file}")
                self.test_data_len, test_tensor_data, info = torch.load(data_file)
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
                torch.save([self.test_data_len, test_tensor_data, info], data_file)

            self.test_data = self.processor.get_batch_data(data=test_tensor_data,
                                                           mode="test",
                                                           batch_size=self.test_batch_size)

            self.test_batch_len = len(self.test_data)
            self.logger.info(f"batch number: {self.test_batch_len}")

        lab_classes = len(self.lab2id)
        rel_classes = len(self.rel2id)
        self.model = DMGModel(args=self.args, lab_classes=lab_classes, rel_classes=rel_classes, device=self.device)

    def get_logits(self, model, batch_data, mode):
        model.eval()
        use_obj = tqdm(self.processor.iter_batch(batch_data, shuffle=False))
        data = []
        torch.cuda.empty_cache()
        for step, batch in enumerate(use_obj, start=1):
            input_ids, offsets, ent_labs, graphs, tri_labs, rel_labs, labs, info, seq_lens = batch
            input_ids = input_ids.to(self.device)
            ent_labs = ent_labs.to(self.device)
            if type(offsets) != list:
                offsets = offsets.to(self.device)
            attention_mask = (input_ids > 0).long()
            graphs = graphs.to(self.device)
            with torch.no_grad():
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                          'offsets': offsets, "ent_labs": ent_labs, "seq_lens": seq_lens, "graphs": graphs}
                outputs = model(**inputs)
                data.append(outputs["tri"])
        torch.save(data, f"./experiments/{self.args.task_name}/{mode}_triggers.pt")
        return data

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

        if self.args.stage == "argument":
            # pretrained_dict = torch.load('save_model/bert_best_model_82.23_stak_gcn.bin')
            pretrained_dict = torch.load(f'experiments/{self.args.task_name}/best_trigger_model.bin')
            model_dict = model.state_dict()
            # 1. filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
            # 2. overwrite entries in the existing state dict
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            if not os.path.exists(f"experiments/{self.args.task_name}/devel_triggers.pt") or not os.path.exists(
                    f"experiments/{self.args.task_name}/devel_triggers.pt"):
                devel_logits = self.get_logits(model=model, batch_data=self.devel_data, mode="devel")
                test_logits = self.get_logits(model=model, batch_data=self.test_data, mode="test")
            else:
                devel_logits = torch.load(f"./experiments/{self.args.task_name}/devel_triggers.pt")
                test_logits = torch.load(f"./experiments/{self.args.task_name}/test_triggers.pt")
        else:
            devel_logits, train_logits = None, None

        loss_cr_fun = nn.CrossEntropyLoss()
        loss_br_fun = nn.CrossEntropyLoss()

        best_epoch, best_f1 = 0.0, 0.0

        for epoch in range(self.epoch):
            torch.cuda.empty_cache()
            model.train()
            self.logger.info(f"staring training epoch: {epoch + 1}")
            all_loss, all_ent_loss, all_arg_loss, step = 0.0, 0.0, 0.0, 1
            use_obj = tqdm(self.processor.iter_batch(self.train_data, shuffle=True))
            t_losses, r_losses, s_losses = 0.0, 0.0, 0.0
            for step, batch in enumerate(use_obj, start=1):
                input_ids, offsets, ent_labs, graphs, tri_labs, rel_labs, labs, info, seq_lens = batch
                input_ids = input_ids.to(self.device)
                ent_labs = ent_labs.to(self.device)
                tri_labs = tri_labs.to(self.device)
                rel_labs = rel_labs.to(self.device)
                labs = labs.to(self.device)
                seq_lens = seq_lens.to(self.device)
                if type(offsets) != list:
                    offsets = offsets.to(self.device)
                attention_mask = (input_ids > 0).long()
                graphs = graphs.to(self.device)
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                          'offsets': offsets, "ent_labs": ent_labs, "seq_lens": seq_lens, "graphs": graphs}

                logits = model(**inputs)

                batch_tri_loss = loss_cr_fun(logits["tri"].reshape(-1, len(self.lab2id)), tri_labs.reshape(-1, ))
                batch_rel_loss = loss_br_fun(logits["rel"].float().reshape(-1, len(self.rel2id)),
                                             rel_labs.reshape(-1, ))
                batch_lab_loss = loss_br_fun(logits["lab"].float(), labs)

                if self.args.stage == "trigger":
                    batch_loss = batch_tri_loss
                elif self.args.stage == "argument":
                    batch_loss = batch_rel_loss + batch_lab_loss
                else:
                    batch_loss = batch_tri_loss + batch_rel_loss + batch_lab_loss

                optimizer.zero_grad()
                batch_loss.backward()

                clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                optimizer.step()
                scheduler.step()
                all_loss += batch_loss.item()
                t_losses += batch_tri_loss.item()
                r_losses += batch_rel_loss.item()
                s_losses += batch_lab_loss.item()

                data_time = str(datetime.datetime.now())[:-3]
                use_obj.set_description(
                    "{0} - [\tINFO] - INFO - loss: {1:.4f}: t-loss: {2:.4f}, r-loss: {3:.4f}, s-loss: {4:.4f}".format(
                        data_time,
                        all_loss / step,
                        t_losses / step,
                        r_losses / step,
                        s_losses / step))

            self.logger.info("overall-loss: {0:.4f}".format(all_loss / step))
            self.logger.info("cross_entropy-t-loss: {0:.4f}".format(t_losses / step))
            self.logger.info("cross_entropy-r-loss: {0:.4f}".format(r_losses / step))
            self.logger.info("cross_entropy-s-loss: {0:.4f}".format(s_losses / step))

            tri_info, rel_info = self.evaluate(batch_data=self.devel_data, model=model, logits=devel_logits)
            p_t, r_t, f_t, y_trigger_pred, y_trigger_gold, y_trigger_std = tri_info
            p_r, r_r, f_r, y_relation_pred, y_relation_gold, y_relation_std = rel_info

            self.logger.info(f"devel metric".center(30, "*"))
            self.logger.info("precision (Trigger): {0:>.4%}".format(p_t))
            self.logger.info("recall (Trigger): {0:>.4%}".format(r_t))
            self.logger.info("f1 (Trigger): {0:>.4%}".format(f_t))

            self.logger.info("precision (Relation): {0:>.4%}".format(p_r))
            self.logger.info("recall (Relation): {0:>.4%}".format(r_r))
            self.logger.info("f1 (Relation): {0:>.4%}".format(f_r))

            self.logger.info(f"*" * 30)
            if self.args.stage == "trigger":
                cur_f = f_t
            else:
                cur_f = f_r
            if best_f1 < cur_f:
                if self.args.stage == "trigger":
                    best_f1 = f_t
                else:
                    best_f1 = f_r

                with open(f"./experiments/{self.args.task_name}/devel_trigger_pred.json", encoding="utf-8",
                          mode="w") as f:
                    json.dump(list(y_trigger_pred), f, indent=4, ensure_ascii=False)

                with open(f"./experiments/{self.args.task_name}/devel_trigger_gold.json", encoding="utf-8",
                          mode="w") as f:
                    json.dump(list(y_trigger_gold), f, indent=4, ensure_ascii=False)

                with open(f"./experiments/{self.args.task_name}/devel_relation_pred.json", encoding="utf-8",
                          mode="w") as f:
                    json.dump(list(y_relation_pred), f, indent=4, ensure_ascii=False)

                with open(f"./experiments/{self.args.task_name}/devel_relation_gold.json", encoding="utf-8",
                          mode="w") as f:
                    json.dump(list(y_relation_gold), f, indent=4, ensure_ascii=False)

                best_epoch = epoch + 1
                torch.save(model.state_dict(),
                           f"./experiments/{self.args.task_name}/best_{self.args.stage}_model.bin")
            self.logger.info("current best f1: tri: {0:>.4%}".format(best_f1))
            self.logger.info(f"current best epoch: {best_epoch}")

            if abs(best_epoch - epoch) > self.args.early_stop:
                self.logger.info("early stop!")
                break

    def evaluate(self, batch_data, model, logits=None):
        torch.cuda.empty_cache()
        model.eval()
        use_obj = tqdm(self.processor.iter_batch(batch_data, shuffle=False))
        y_trigger_pred, y_trigger_gold, y_trigger_std = set(), set(), set()
        y_relation_pred, y_relation_gold, y_relation_std = set(), set(), set()
        for step, batch in enumerate(use_obj, start=1):
            input_ids, offsets, ent_labs, graphs, tri_labs, rel_labs, labs, info, seq_lens = batch
            input_ids = input_ids.to(self.device)
            ent_labs = ent_labs.to(self.device)
            if type(offsets) != list:
                offsets = offsets.to(self.device)
            attention_mask = (input_ids > 0).long()
            graphs = graphs.to(self.device)
            with torch.no_grad():
                inputs = {'input_ids': input_ids, 'attention_mask': attention_mask,
                          'offsets': offsets, "ent_labs": ent_labs, "seq_lens": seq_lens, "graphs": graphs}
                outputs = model(**inputs)
            if self.args.stage == "argument":
                outputs["tri"] = logits[step - 1]
            batch_triggers, gold_triggers, batch_stp_triggers, batch_p_relations, batch_g_relations, batch_stp_relations = generation(
                outputs, info, id2tri=self.id2lab,
                id2rel=self.id2rel,
                tag2id=self.tag2id, prior=self.prior, struct=self.struct)
            y_trigger_pred.update(batch_triggers)
            y_trigger_gold.update(gold_triggers)
            y_trigger_std.update(batch_stp_triggers)

            y_relation_pred.update(batch_p_relations)
            y_relation_gold.update(batch_g_relations)

        p, r, f = performance_report(y_true=y_trigger_gold, y_pred=y_trigger_pred)
        p1, r1, f1 = performance_report(y_true=y_relation_gold, y_pred=y_relation_pred)

        return [p, r, f, y_trigger_pred, y_trigger_gold, y_trigger_std], [p1, r1, f1, y_relation_pred, y_relation_gold,
                                                                          y_relation_std]

    def prediction(self):
        test_logits = torch.load(f"./experiments/{self.args.task_name}/test_triggers.pt")
        self.model.load_state_dict(torch.load(f"./experiments/{self.args.task_name}/best_argument_model.bin"))
        self.model.cuda()
        self.model.eval()

        tri_info, rel_info = self.evaluate(batch_data=self.test_data, model=self.model, logits=test_logits)
        p_t, r_t, f_t, y_trigger_pred, y_trigger_gold, y_trigger_std = tri_info
        p_r, r_r, f_r, y_relation_pred, y_relation_gold, y_relation_std = rel_info
        with open(f"./experiments/{self.args.task_name}/test_trigger_pred.json", encoding="utf-8", mode="w") as f:
            json.dump(list(y_trigger_pred), f, indent=4, ensure_ascii=False)

        with open(f"./experiments/{self.args.task_name}/test_trigger_gold.json", encoding="utf-8", mode="w") as f:
            json.dump(list(y_trigger_gold), f, indent=4, ensure_ascii=False)

        with open(f"./experiments/{self.args.task_name}/test_relation_pred.json", encoding="utf-8", mode="w") as f:
            json.dump(list(y_relation_pred), f, indent=4, ensure_ascii=False)

        with open(f"./experiments/{self.args.task_name}/test_relation_gold.json", encoding="utf-8", mode="w") as f:
            json.dump(list(y_relation_gold), f, indent=4, ensure_ascii=False)

        self.logger.info(f"test metric".center(30, "*"))

        self.logger.info("precision (Trigger): {0:>.4%}".format(p_t))
        self.logger.info("recall (Trigger): {0:>.4%}".format(r_t))
        self.logger.info("f1 (Trigger): {0:>.4%}".format(f_t))

        self.logger.info("precision (Relation): {0:>.4%}".format(p_r))
        self.logger.info("recall (Relation): {0:>.4%}".format(r_r))
        self.logger.info("f1 (Relation): {0:>.4%}".format(f_r))

        self.logger.info(f"*" * 30)


if __name__ == '__main__':
    task_names = ["MLEE", "GE13"]
    task_name = "GE13"
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--task_name', type=str, default=task_name,
                        choices=task_names)
    parser.add_argument('--train_path', type=str, default=f"./sources/{task_name}/std-train.json")
    parser.add_argument('--devel_path', type=str, default=f"./sources/{task_name}/std-devel.json")
    parser.add_argument('--test_path', type=str, default=f"./sources/{task_name}/std-test.json")
    parser.add_argument('--prior_path', type=str, default=f"./sources/{task_name}/prior.json")
    parser.add_argument('--tag_path', type=str, default=f"./sources/{task_name}/tag_limits.json")

    parser.add_argument('--output_path', type=str, default=f"./experiments/{task_name}")
    parser.add_argument('--bert_path', type=str, default="./bert/scibert_scivocab_uncased")  # 本地不能用

    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--early_stop', type=int, default=50)
    parser.add_argument('--ent_dim', type=int, default=50)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--adam_epsilon', type=float, default=1e-8)
    parser.add_argument('--warmup_proportion', type=float, default=0.0)
    parser.add_argument('--drop_rate', type=float, default=0.5)

    parser.add_argument('--stage', type=str, default="trigger", choices=["trigger", "argument"])

    parser.add_argument('--gpu_id', type=int, default=0)

    parser.add_argument('--train_batch_size', type=int, default=8)
    parser.add_argument('--devel_batch_size', type=int, default=8)
    parser.add_argument('--test_batch_size', type=int, default=8)

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
    # Pipeline.run_train()
    Pipeline.prediction()
