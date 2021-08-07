import torch
import torch.nn as nn
import torch
from torch.autograd import Variable
import copy
from transformers.modeling_bert import BertLayerNorm
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, MSELoss
# from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
#                           BertConfig, BertForMaskedLM, BertTokenizer,
#                           GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
#                           OpenAIGPTConfig, OpenAIGPTLMHeadModel, OpenAIGPTTokenizer,
#                           RobertaConfig, RobertaModel, RobertaTokenizer,
#                           DistilBertConfig, DistilBertForMaskedLM, DistilBertTokenizer)
from transformers.modeling_utils import PreTrainedModel


class ModelBinary(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelBinary, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 1))
        loss = self.loss_func(logits, labels.float())
        predictions = (logits > 0.5).int()  # (Batch, )
        return loss, predictions


class ModelContra(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelContra, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        nl_vec = nl_vec.unsqueeze(1).repeat([1, bs, 1])
        code_vec = code_vec.unsqueeze(0).repeat([bs, 1, 1])
        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 2)).squeeze(2) # (Batch, Batch)
        matrix_labels = torch.diag(labels).float()  # (Batch, Batch)
        poss = logits[matrix_labels==1]
        negs = logits[matrix_labels==0]

        # loss = self.loss_func(logits, matrix_labels)
        # bce equals to -(torch.log(1-logits[matrix_labels==0]).sum() + torch.log(logits[matrix_labels==1]).sum()) / (bs*bs)
        loss = - (torch.log(1 - negs).mean() + torch.log(poss).mean())
        predictions = (logits.gather(0, torch.arange(bs, device=loss.device).unsqueeze(0)).squeeze(0) > 0.5).int()
        return loss, predictions

class ModelContraOnline(PreTrainedModel):
    def __init__(self, encoder, config, tokenizer, args):
        super(ModelContraOnline, self).__init__(config)
        self.encoder = encoder
        self.config = config
        self.tokenizer = tokenizer
        self.mlp = nn.Sequential(nn.Linear(768*4, 768),
                                 nn.Tanh(),
                                 nn.Linear(768, 1),
                                 nn.Sigmoid())
        self.loss_func = nn.BCELoss()
        self.args = args

    def forward(self, code_inputs, nl_inputs, labels, return_vec=False):
        bs = code_inputs.shape[0]
        inputs = torch.cat((code_inputs, nl_inputs), 0)
        outputs = self.encoder(inputs, attention_mask=inputs.ne(1))[1]
        code_vec = outputs[:bs]
        nl_vec = outputs[bs:]
        if return_vec:
            return code_vec, nl_vec

        nl_vec = nl_vec.unsqueeze(1).repeat([1, bs, 1])
        code_vec = code_vec.unsqueeze(0).repeat([bs, 1, 1])
        logits = self.mlp(torch.cat((nl_vec, code_vec, nl_vec-code_vec, nl_vec*code_vec), 2)).squeeze(2) # (Batch, Batch)
        matrix_labels = torch.diag(labels).float()  # (Batch, Batch)
        poss = logits[matrix_labels==1]
        negs = logits[matrix_labels==0]

        negative_pairs = negs[negs < (poss.max() if len(poss) > 1 else negs.mean())]
        positive_pairs = poss[poss > (negs.min() if len(negs) > 1 else poss.mean())]

        loss = - (torch.log(1 - negative_pairs).mean() + torch.log(positive_pairs).mean())
        predictions = (logits.gather(0, torch.arange(bs, device=loss.device).unsqueeze(0)).squeeze(0) > 0.5).int()
        return loss, predictions

