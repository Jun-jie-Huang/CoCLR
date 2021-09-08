from __future__ import absolute_import, division, print_function

import argparse
import glob
import logging
import os
import pickle
import random
import re
import shutil

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler,TensorDataset
from torch.utils.data.distributed import DistributedSampler
import json
try:
    from torch.utils.tensorboard import SummaryWriter
except:
    from tensorboardX import SummaryWriter

from tqdm import tqdm, trange
import multiprocessing
from models import ModelBinary, ModelContra, ModelContraOnline
from utils import acc_and_f1
cpu_cont = multiprocessing.cpu_count()
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          BertConfig, BertForMaskedLM, BertTokenizer,
                          GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
                          OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer,
                          DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)

logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    'gpt2': (GPT2Config, GPT2LMHeadModel, GPT2Tokenizer),
    'openai-gpt': (OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer),
    'bert': (BertConfig, BertForMaskedLM, BertTokenizer),
    'roberta': (RobertaConfig, RobertaModel, RobertaTokenizer),
    'distilbert': (DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
}


class InputFeatures(object):
    """A single training/test features for a example."""
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, label, idx):
        self.code_tokens = code_tokens
        self.code_ids = code_ids        # code tokenize后的ids
        self.nl_tokens = nl_tokens
        self.nl_ids = nl_ids            # nl tokenize后的ids
        self.label = label
        self.idx = idx


class InputFeaturesTrip(InputFeatures):
    """A single training/test features for a example. Add docstring seperately. """
    def __init__(self, code_tokens, code_ids, nl_tokens, nl_ids, ds_tokens, ds_ids, label, idx):
        super(InputFeaturesTrip, self).__init__(code_tokens, code_ids, nl_tokens, nl_ids, label, idx)
        self.ds_tokens = ds_tokens
        self.ds_ids = ds_ids


def convert_examples_to_features(js, tokenizer, args):
    # label
    label = js['label']

    # code
    code = js['code']
    if args.code_type == 'code_tokens':
        code = js['code_tokens']
    # code = js['code']
    # code = js['tokens_combine']
    code_tokens = tokenizer.tokenize(code)[:args.max_seq_length-2]
    code_tokens = [tokenizer.cls_token]+code_tokens+[tokenizer.sep_token]
    code_ids = tokenizer.convert_tokens_to_ids(code_tokens)
    padding_length = args.max_seq_length - len(code_ids)
    code_ids += [tokenizer.pad_token_id]*padding_length

    nl = js['doc']
    # nl = js['query']
    nl_tokens = tokenizer.tokenize(nl)[:args.max_seq_length-2]
    nl_tokens = [tokenizer.cls_token]+nl_tokens+[tokenizer.sep_token]
    nl_ids = tokenizer.convert_tokens_to_ids(nl_tokens)
    padding_length = args.max_seq_length - len(nl_ids)
    nl_ids += [tokenizer.pad_token_id]*padding_length

    # ds = js['docstring_tokens']
    # ds_tokens = tokenizer.tokenize(ds)[:args.max_seq_length-2]
    # ds_tokens = [tokenizer.cls_token]+ds_tokens+[tokenizer.sep_token]
    # ds_ids = tokenizer.convert_tokens_to_ids(ds_tokens)
    # padding_length = args.max_seq_length - len(ds_ids)
    # ds_ids += [tokenizer.pad_token_id]*padding_length

    # return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, label, js['idx'])
    return InputFeatures(code_tokens, code_ids, nl_tokens, nl_ids, label, js['idx'])


class TextDataset(Dataset):
    def __init__(self, tokenizer, args, file_path=None):
        # file：json文件，每一个dict中包括：idx, query, doc, code(或者叫function_tokens，list形式), docstring_tokens(list形式)
        self.examples = []
        self.data=[]
        with open(file_path, 'r') as f:
            self.data = json.load(f)
        for js in self.data:
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        # 如果是training集，就print前3个
        if 'train' in file_path:
            for idx, example in enumerate(self.examples[:3]):
                logger.info("*** Example ***")
                logger.info("idx: {}".format(idx))
                logger.info("code_tokens: {}".format([x.replace('\u0120','_') for x in example.code_tokens]))
                logger.info("code_ids: {}".format(' '.join(map(str, example.code_ids))))
                logger.info("nl_tokens: {}".format([x.replace('\u0120','_') for x in example.nl_tokens]))
                logger.info("nl_ids: {}".format(' '.join(map(str, example.nl_ids))))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return torch.tensor(self.examples[i].code_ids), \
               torch.tensor(self.examples[i].nl_ids),\
               torch.tensor(self.examples[i].label),
               # torch.tensor(i)

class RetievalDataset(Dataset):
    def __init__(self, tokenizer, args, data_path=None):
        self.codes = []
        self.data = []
        self.examples = []  # codebase用code和code当成6k pair; testdata用query和code当成pair来算; examples前0-6267，code为6267-6767
        code_file = args.retrieval_code_base
        data_file = data_path
        with open(code_file, 'r') as f:
            self.codes = json.loads(f.read())
        for code, code_id in self.codes.items():
            js = {'code': code, 'doc': code, 'label': 1, 'idx': code_id}
            self.examples.append(convert_examples_to_features(js, tokenizer, args))
        with open(data_file, 'r') as f:
            self.data = json.load(f)
        for js in self.data:
            new_js = {'code': js['code'], 'doc': js['doc'], 'label': js['label'], 'idx': js['retrieval_idx']}
            self.examples.append(convert_examples_to_features(new_js, tokenizer, args))

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        """ return both tokenized code ids and nl ids and label"""
        return torch.tensor(self.examples[i].code_ids), \
               torch.tensor(self.examples[i].nl_ids),\
               torch.tensor(self.examples[i].label)


def set_seed(seed=45):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True



def train(args, train_dataset, model, tokenizer):
    """ Train the model """

    args.train_batch_size = args.per_gpu_train_batch_size * max(1, args.n_gpu)
    train_sampler = RandomSampler(train_dataset) if args.local_rank == -1 else DistributedSampler(train_dataset)
    train_dataloader = DataLoader(train_dataset, sampler=train_sampler, batch_size=args.train_batch_size, num_workers=4, pin_memory=True)

    args.save_steps = len(train_dataloader) if args.save_steps<=0 else args.save_steps
    args.warmup_steps = len(train_dataloader) if args.warmup_steps<=0 else args.warmup_steps
    args.logging_steps = len(train_dataloader)

    if args.max_steps > 0:
        t_total = args.max_steps
        args.num_train_epochs = args.max_steps // (len(train_dataloader) // args.gradient_accumulation_steps)
    else:
        t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.num_train_epochs

    # Prepare optimizer and schedule (linear warmup and decay)
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, args.warmup_steps, t_total)

    model.to(args.device)
    if args.fp16:
        try:
            from apex import amp
        except ImportError:
            raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
        model, optimizer = amp.initialize(model, optimizer, opt_level=args.fp16_opt_level)

    # multi-gpu training (should be after apex fp16 initialization)
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Distributed training (should be after apex fp16 initialization)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                          output_device=args.local_rank,
                                                          find_unused_parameters=True)

    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    # scheduler_last = os.path.join(checkpoint_last, 'scheduler.pt')
    # optimizer_last = os.path.join(checkpoint_last, 'optimizer.pt')
    # if os.path.exists(scheduler_last):
    #     scheduler.load_state_dict(torch.load(scheduler_last))
    # if os.path.exists(optimizer_last):
    #     optimizer.load_state_dict(torch.load(optimizer_last))

    # Train!
    logger.info("***** Running training *****")
    logger.info("  Num examples = %d", len(train_dataset))
    logger.info("  Num Epochs = %d", args.num_train_epochs)
    logger.info("  Instantaneous batch size per GPU = %d", args.per_gpu_train_batch_size)
    logger.info("  Total train batch size (w. parallel, distributed & accumulation) = %d",
                args.train_batch_size * args.gradient_accumulation_steps * (
                    torch.distributed.get_world_size() if args.local_rank != -1 else 1))
    logger.info("  Gradient Accumulation steps = %d", args.gradient_accumulation_steps)
    logger.info("  Total optimization steps = %d", t_total)

    global_step = args.start_step
    tr_loss, logging_loss, avg_loss, tr_nb, tr_num, train_loss = 0.0, 0.0, 0.0, 0, 0, 0
    best_mrr = 0.0
    # best_results = {"acc": -1.0, "precision": -1.0, "recall": -1.0, "f1": -1.0, "acc_and_f1": -1.0}
    best_results = {"acc": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "acc_and_f1": 0.0, "mrr": 0.0}
    # model.resize_token_embeddings(len(tokenizer))
    model.zero_grad()
    logger.info(model)

    for idx in range(args.start_epoch, int(args.num_train_epochs)):
        bar = tqdm(enumerate(train_dataloader))
        tr_num=0
        train_loss=0
        for step, batch in bar:
            code_inputs = batch[0].to(args.device)
            nl_inputs = batch[1].to(args.device)
            # ds_inputs = batch[2].to(args.device)
            labels = batch[2].to(args.device)

            model.train()
            loss, predictions = model(code_inputs, nl_inputs, labels)

            if args.n_gpu > 1:
                loss = loss.mean()  # mean() to average on multi-gpu parallel training
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            if args.fp16:
                with amp.scale_loss(loss, optimizer) as scaled_loss:
                    scaled_loss.backward()
                torch.nn.utils.clip_grad_norm_(amp.master_params(optimizer), args.max_grad_norm)
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            tr_loss += loss.item()
            tr_num += 1
            train_loss += loss.item()
            if avg_loss == 0:
                avg_loss = tr_loss
            avg_loss = round(train_loss/tr_num, 5)
            bar.set_description("epoch {} step {} loss {}".format(idx, step+1, avg_loss))

            # logger.info("\nstep:{}, lb_sum:{}, index:{}".format(step+1, torch.sum(labels), batch[3]))
            # logger.info(", ".join([train_dataset.data[i.tolist()]['idx'] for i in batch[3]]))

            if (step + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
                global_step += 1
                avg_loss = round(np.exp((tr_loss - logging_loss) / (global_step - tr_nb)), 4)
                if args.local_rank in [-1, 0] and args.logging_steps > 0 and global_step % args.logging_steps == 0:
                    logging_loss = tr_loss
                    tr_nb = global_step

                if args.local_rank in [-1, 0] and args.save_steps > 0 and global_step % args.save_steps == 0:

                    if args.local_rank == -1 and args.evaluate_during_training:  # Only evaluate when single GPU otherwise metrics may not average well
                        mrr = evaluate(args, model, tokenizer, eval_when_training=True)
                        logger.info(" Mrr = %s", round(mrr, 4))
                        # Save model checkpoint
                        if mrr >= best_results['mrr']:
                            best_results['mrr'] = mrr

                            # save
                            checkpoint_prefix = 'checkpoint-best-mrr'
                            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                            if not os.path.exists(output_dir):
                                os.makedirs(output_dir)
                            model_to_save = model.module if hasattr(model, 'module') else model

                            torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                            tokenizer.save_pretrained(output_dir)
                            torch.save(args, os.path.join(output_dir, 'training_{}.bin'.format(idx)))
                            logger.info("Saving model checkpoint to %s", output_dir)
                            # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                            # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                            # logger.info("Saving optimizer and scheduler states to %s", output_dir)

                            # torch.save(model_to_save.state_dict(), output_dir)
                            # logger.info("Saving model checkpoint to %s", output_dir)

                    if args.local_rank == -1:
                        checkpoint_prefix = 'checkpoint-last'
                        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        model_to_save = model.module if hasattr(model, 'module') else model
                        torch.save(model_to_save.state_dict(), os.path.join(output_dir, 'pytorch_model.bin'))
                        tokenizer.save_pretrained(output_dir)

                        idx_file = os.path.join(output_dir, 'idx_file.txt')
                        with open(idx_file, 'w', encoding='utf-8') as idxf:
                            idxf.write(str(args.start_epoch + idx) + '\n')
                        # logger.info("Saving model checkpoint to %s", output_dir)
                        # torch.save(optimizer.state_dict(), os.path.join(output_dir, "optimizer.pt"))
                        # torch.save(scheduler.state_dict(), os.path.join(output_dir, "scheduler.pt"))
                        # logger.info("Saving optimizer and scheduler states to %s", output_dir)
                        step_file = os.path.join(output_dir, 'step_file.txt')
                        with open(step_file, 'w', encoding='utf-8') as stepf:
                            stepf.write(str(global_step) + '\n')
                        # model_to_save = model.module if hasattr(model, 'module') else model
                        # output_dir = os.path.join(output_dir, '{}'.format('model.bin'))
                        # torch.save(model_to_save.state_dict(), output_dir)
                        # logger.info("Saving model checkpoint to %s", output_dir)


eval_dataset=None
def evaluate(args, model, tokenizer,eval_when_training=False):
    """ dev集上时，应该使用一样的计算方法 """
    eval_output_dir = args.output_dir
    global eval_dataset
    if eval_dataset is None:
        args.retrieval_code_base = os.path.join(args.data_dir, args.retrieval_code_base)
        eval_data_path = os.path.join(args.data_dir, args.eval_data_file)
        eval_dataset = RetievalDataset(tokenizer, args, eval_data_path)

    if not os.path.exists(eval_output_dir) and args.local_rank in [-1, 0]:
        os.makedirs(eval_output_dir)

    args.eval_batch_size = args.per_gpu_retrieval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(eval_dataset) if args.local_rank == -1 else DistributedSampler(eval_dataset)
    eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size, num_workers=4, pin_memory=True)

    # multi-gpu evaluate
    if args.n_gpu > 1 and eval_when_training is False:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num examples = %d", len(eval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_loss = 0.0
    nb_eval_steps = 0
    model.eval()
    all_first_vec = []
    all_second_vec = []
    for batch in eval_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        with torch.no_grad():
            code_vec, nl_vec = model(code_inputs, nl_inputs, labels, return_vec=True)
            all_first_vec.append(code_vec.cpu())
            all_second_vec.append(nl_vec.cpu())
        nb_eval_steps += 1
    code_vectors = torch.cat(all_first_vec, 0).squeeze()[:6267, :]
    # print(code_vectors.size())
    query_vectors = torch.cat(all_second_vec, 0).squeeze()[6267:, :]
    # print(query_vectors.size())
    assert(code_vectors.size(0)==6267)
    assert(query_vectors.size(0)==500)

    repeat_nl_vec = query_vectors.unsqueeze(1).to(args.device)
    scores = []
    with torch.no_grad():
        for idx in trange(6267):
            repeat_code_vec = code_vectors[idx, :].unsqueeze(0).repeat([500, 1, 1]).to(args.device)
            logits = model.mlp(torch.cat((repeat_nl_vec, repeat_code_vec, repeat_nl_vec - repeat_code_vec, repeat_nl_vec * repeat_code_vec),2)).squeeze(2)
            scores.append(logits.cpu())

    scores = torch.cat(scores, 1)
    results = []
    mrr = 0
    for idx in range(len(scores)):
        rank = torch.argsort(-scores[idx]).tolist()
        # logger.info(rank[:10])
        example = eval_dataset.examples[idx+6267]
        # logger.info(example.idx)
        item = {}
        item['ans'] = example.idx
        item['rank'] = rank
        item['rr'] = 1 / (rank.index(example.idx)+1)
        mrr += item['rr']
        results.append(item)
    mrr = mrr / len(scores)
    logger.info("  Final test MRR {}".format(mrr))
    return mrr


def retrieval(args, model, tokenizer):
    if not args.retrieval_predictions_output:
        args.retrieval_predictions_output = os.path.join(args.output_dir, 'retrieval_outputs.txt')
    if not os.path.exists(os.path.dirname(args.retrieval_predictions_output)):
        os.makedirs(os.path.dirname(args.retrieval_predictions_output))

    args.retrieval_code_base = os.path.join(args.data_dir, args.retrieval_code_base)
    args.test_data_file = os.path.join(args.data_dir, args.test_data_file)
    retrieval_dataset = RetievalDataset(tokenizer, args, args.test_data_file)

    args.eval_batch_size = args.per_gpu_retrieval_batch_size * max(1, args.n_gpu)
    # Note that DistributedSampler samples randomly
    eval_sampler = SequentialSampler(retrieval_dataset) if args.local_rank == -1 else DistributedSampler(retrieval_dataset)
    eval_dataloader = DataLoader(retrieval_dataset, sampler=eval_sampler, batch_size=args.eval_batch_size)

    # multi-gpu evaluate
    if args.n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # Eval!
    logger.info("***** Running Test *****")
    logger.info("  Num examples = %d", len(retrieval_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    eval_loss = 0.0
    nb_eval_steps = 0
    all_first_vec = []
    all_second_vec = []
    for batch in eval_dataloader:
        code_inputs = batch[0].to(args.device)
        nl_inputs = batch[1].to(args.device)
        labels = batch[2].to(args.device)
        with torch.no_grad():
            code_vec, nl_vec = model(code_inputs, nl_inputs, labels, return_vec=True)
            all_first_vec.append(code_vec.cpu())
            all_second_vec.append(nl_vec.cpu())
        nb_eval_steps += 1
    code_vectors = torch.cat(all_first_vec, 0).squeeze()[:6267, :]
    print(code_vectors.size())
    query_vectors = torch.cat(all_second_vec, 0).squeeze()[6267:, :]
    print(query_vectors.size())
    assert(code_vectors.size(0)==6267)
    assert(query_vectors.size(0)==500)

    repeat_nl_vec = query_vectors.unsqueeze(1).to(args.device)
    scores = []
    with torch.no_grad():
        for idx in trange(6267):
            repeat_code_vec = code_vectors[idx, :].unsqueeze(0).repeat([500, 1, 1]).to(args.device)
            logits = model.mlp(torch.cat((repeat_nl_vec, repeat_code_vec, repeat_nl_vec - repeat_code_vec, repeat_nl_vec * repeat_code_vec),2)).squeeze(2)
            scores.append(logits.cpu())

    scores = torch.cat(scores, 1)
    results = []
    mrr = 0
    for idx in range(len(scores)):
        rank = torch.argsort(-scores[idx]).tolist()
        # logger.info(rank[:10])
        example = retrieval_dataset.examples[idx+6267]
        # logger.info(example.idx)
        item = {}
        item['ans'] = example.idx
        item['rank'] = rank
        item['rr'] = 1 / (rank.index(example.idx)+1)
        mrr += item['rr']
        results.append(item)
    mrr = mrr / len(scores)
    logger.info("  Final test MRR {}".format(mrr))
    with open(args.retrieval_predictions_output,'w') as f:
        json.dump(results, f, indent=1)
    return mrr


def main():
    parser = argparse.ArgumentParser()

    ## Required parameters
    parser.add_argument("--data_dir", default=None, type=str, required=True,
                        help="The input data dir. Should contain the .tsv files (or other data files) for the task.")
    parser.add_argument("--train_data_file", default=None, type=str,
                        help="The input training data file (a text file).")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")

    ## Other parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--test_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the perplexity on (a text file).")
    parser.add_argument("--retrieval_code_base", default=None, type=str,
                        help="An optional input retrieval evaluation data file for all code bases.")

    parser.add_argument("--model_type", default="roberta", type=str,
                        help="The model architecture to be fine-tuned.")
    parser.add_argument("--augment", action='store_true', help="Whether use QRA and IBA augmentation or not.")
    parser.add_argument("--encoder_name_or_path", default=None, type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--checkpoint_path", default=None, type=str,
                        help="The checkpoint path of model to continue training.")

    parser.add_argument("--config_name", default="", type=str,
                        help="Optional pretrained config name or path if not the same as encoder_name_or_path")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Optional pretrained tokenizer name or path if not the same as encoder_name_or_path")
    parser.add_argument("--cache_dir", default="", type=str,
                        help="Optional directory to store the pre-trained models downloaded from s3 (instread of the default one)")
    parser.add_argument("--max_seq_length", default=-1, type=int,
                        help="Optional input sequence length after tokenization."
                             "The training dataset will be truncated in block of this size for training."
                             "Default to the model max input length for single sentence inputs (take into account special tokens).")
    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run training.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_retrieval", action='store_true',
                        help="Whether to run test on the dev/test set and do code retrieval.")
    parser.add_argument("--evaluate_during_training", action='store_true',
                        help="Run evaluation during training at each logging step.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")

    parser.add_argument("--per_gpu_train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--per_gpu_eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--per_gpu_retrieval_batch_size", default=67, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument("--mrr_rank", default=100, type=int,
                        help="Number of retrieved item for computing mrr")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--num_train_epochs", default=3, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--code_type", default='code', type=str,
                        help='use `code` or `code_tokens` in the json file to index.')

    parser.add_argument('--logging_steps', type=int, default=50,
                        help="Log every X updates steps.")
    parser.add_argument('--save_steps', type=int, default=0,
                        help="Save checkpoint every X updates steps.")
    parser.add_argument('--save_total_limit', type=int, default=None,
                        help='Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default')
    parser.add_argument("--eval_all_checkpoints", action='store_true',
                        help="Evaluate all checkpoints starting with the same prefix as encoder_name_or_path ending and ending with step number")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")
    parser.add_argument('--overwrite_output_dir', action='store_true',
                        help="Overwrite the content of the output directory")
    parser.add_argument('--overwrite_cache', action='store_true',
                        help="Overwrite the cached training and evaluation sets")
    parser.add_argument('--seed', type=int, default=45,
                        help="random seed for initialization")
    parser.add_argument('--fp16', action='store_true',
                        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit")
    parser.add_argument('--fp16_opt_level', type=str, default='O1',
                        help="For fp16: Apex AMP optimization level selected in ['O0', 'O1', 'O2', and 'O3']."
                             "See details at https://nvidia.github.io/apex/amp.html")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--server_ip', type=str, default='', help="For distant debugging.")
    parser.add_argument('--server_port', type=str, default='', help="For distant debugging.")

    parser.add_argument("--pred_model_dir", default=None, type=str,
                        help='model for prediction')
    parser.add_argument("--test_result_dir", default='test_results.tsv', type=str,
                        help='path to store test result')
    parser.add_argument("--test_predictions_output", default=None, type=str,
                        help="The output directory where the model predictions")
    parser.add_argument("--retrieval_predictions_output", default=None, type=str,
                        help="The output directory where the model predictions")
    args = parser.parse_args()


    # Setup distant debugging if needed
    if args.server_ip and args.server_port:
        # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
        import ptvsd
        print("Waiting for debugger attach")
        ptvsd.enable_attach(address=(args.server_ip, args.server_port), redirect_output=True)
        ptvsd.wait_for_attach()

    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    args.device = device
    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1), args.fp16)

    # Set seed
    set_seed(args.seed)

    # Load pretrained model and tokenizer
    if args.local_rank not in [-1, 0]:
        torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training download model & vocab

    args.start_epoch = 0
    args.start_step = 0
    checkpoint_last = os.path.join(args.output_dir, 'checkpoint-last')
    if os.path.exists(checkpoint_last) and os.listdir(checkpoint_last):
        # args.encoder_name_or_path = os.path.join(checkpoint_last, 'pytorch_model.bin')
        args.config_name = os.path.join(checkpoint_last, 'config.json')
        idx_file = os.path.join(checkpoint_last, 'idx_file.txt')
        with open(idx_file, encoding='utf-8') as idxf:
            args.start_epoch = int(idxf.readlines()[0].strip()) + 1

        step_file = os.path.join(checkpoint_last, 'step_file.txt')
        if os.path.exists(step_file):
            with open(step_file, encoding='utf-8') as stepf:
                args.start_step = int(stepf.readlines()[0].strip())

        logger.info("reload model from {}, resume from {} epoch".format(checkpoint_last, args.start_epoch))

    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.encoder_name_or_path,
                                          cache_dir=args.cache_dir if args.cache_dir else None)
    config.num_labels = 2
    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.encoder_name_or_path,
                                                do_lower_case=args.do_lower_case,
                                                cache_dir=args.cache_dir if args.cache_dir else None)
    if args.max_seq_length <= 0:
        args.max_seq_length = tokenizer.max_len_single_sentence  # Our input block size will be the max possible for the model
    args.max_seq_length = min(args.max_seq_length, tokenizer.max_len_single_sentence)
    if args.encoder_name_or_path:
        model = model_class.from_pretrained(args.encoder_name_or_path,
                                            from_tf=bool('.ckpt' in args.encoder_name_or_path),
                                            config=config,
                                            cache_dir=args.cache_dir if args.cache_dir else None)
    else:
        model = model_class(config)

    if args.augment and not args.do_retrieval:
        model = ModelContra(model, config, tokenizer, args)
    else:
        model = ModelBinary(model, config, tokenizer, args)

    if args.checkpoint_path:
        model.load_state_dict(torch.load(os.path.join(args.checkpoint_path, 'pytorch_model.bin')))
    if args.local_rank == 0:
        torch.distributed.barrier()  # End of barrier to make sure only the first process in distributed training download model & vocab

    logger.info("Training/evaluation parameters %s", args)

    # Training
    if args.do_train:
        if args.local_rank not in [-1, 0]:
            torch.distributed.barrier()  # Barrier to make sure only the first process in distributed training process the dataset, and the others will use the cache
        train_data_path = os.path.join(args.data_dir, args.train_data_file)
        train_dataset = TextDataset(tokenizer, args, train_data_path)
        train(args, train_dataset, model, tokenizer)

    # Evaluation
    results = {}
    if args.do_eval and args.local_rank in [-1, 0]:
        checkpoint_prefix = 'checkpoint-best-mrr'
        output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        model.load_state_dict(torch.load(os.path.join(output_dir, 'pytorch_model.bin')))
        tokenizer = tokenizer.from_pretrained(output_dir)
        model.to(args.device)
        mrr = evaluate(args, model, tokenizer)
        logger.info("***** Eval results *****")
        logger.info("  Eval MRR = %s", str(mrr))
        logger.info("Eval Model From: {}".format(os.path.join(output_dir, 'pytorch_model.bin')))
        logger.info("***** Eval results *****")

    if args.do_retrieval and args.local_rank in [-1, 0]:

        logger.info("***** Retrieval results *****")
        checkpoint_prefix = 'checkpoint-best-mrr'
        if checkpoint_prefix not in args.output_dir and \
                os.path.exists(os.path.join(args.output_dir, checkpoint_prefix)):
            output_dir = os.path.join(args.output_dir, checkpoint_prefix)
        else:
            output_dir = args.output_dir
        if not args.pred_model_dir:
            model_path = os.path.join(output_dir, 'pytorch_model.bin')
        else:
            model_path = os.path.join(args.pred_model_dir, 'pytorch_model.bin')
        logger.info(model_path)
        model.load_state_dict(torch.load(model_path))
        tokenizer = tokenizer.from_pretrained(output_dir)
        model.to(args.device)
        mrr = retrieval(args, model, tokenizer)
        logger.info("Test Model From: {}".format(model_path))

    return results


if __name__ == "__main__":
    main()


