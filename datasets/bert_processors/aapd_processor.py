import os
import sys
sys.path.append('./datasets')
import tqdm
import torch.nn as nn
import copy
import torch
import math
from dc1d.nn import DeformConv1d

import torch.nn.functional as F
import numpy as np
from einops import rearrange, repeat
from bert_processors.abstract_processor import BertProcessor, InputExample
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class PositionwiseFeedForward(nn.Module):
     "Implements FFN equation."

     def __init__(self, d_model, d_ff, dropout=0.1):
         super(PositionwiseFeedForward, self).__init__()
         self.w_1 = nn.Linear(d_model, d_ff)
         self.w_2 = nn.Linear(d_ff, d_model)
         self.dropout = nn.Dropout(dropout)

     def forward(self, x):
         return self.w_2(self.dropout(F.relu(self.w_1(x))))

def attention(query, key, value, mask=None,mask1=None,mask2=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    key_ = key.transpose(-2, -1)
    scores = torch.matmul(query, key_) / math.sqrt(d_k)
    if mask1 is not None and mask is not None:
        scores = torch.mul((1 + torch.mul(F.softmax(mask1.masked_fill(mask1 == 0, -1e9), dim=-2), mask1)),scores) * Weight
        scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), scores
    elif mask == None and mask1 is not None:
        scores = torch.mul((1 + torch.mul(F.softmax(mask1.masked_fill(mask1 == 0, -1e9), dim=-2), mask1)),scores) * Weight
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value), scores
    elif mask is not None and mask1 == None:
        scores = scores.masked_fill(mask == 0, -1e9)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value),scores
    else:
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        return torch.matmul(p_attn, value),scores

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        #Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h  # 48=768//16
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.norm = LayerNorm(110)

    def forward(self, query, key, value, mask=None, mask1=None,mask2=None):
        # query,key,value:torch.Size([2, 10, 768])
        # if mask is not None:
        #     # Same mask applied to all h heads.
        #     mask = mask.unsqueeze(1)
        nbatches = query.size(0)    #2
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                             for l, x in zip(self.linears, (query, key, value))]  # query,key,value:torch.Size([30, 8, 10, 64])
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask, mask1=mask1,mask2=mask2, dropout=self.dropout)
         # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(
                  nbatches, -1, self.h * self.d_k)
        ret = self.linears[-1](x)  # torch.Size([2, 10, 768])
        return ret,self.attn
#layer normalization [(cite)](https://arxiv.org/abs/1607.06450). do on

class customizedModule(nn.Module):
    def __init__(self):
        super(customizedModule, self).__init__()

    # linear transformation (w/ initialization) + activation + dropout
    def customizedLinear(self, in_dim, out_dim, activation=None, dropout=False):
        cl = nn.Sequential(nn.Linear(in_dim, out_dim))
        nn.init.xavier_uniform(cl[0].weight)
        nn.init.constant(cl[0].bias, 0)

        if activation is not None:
            cl.add_module(str(len(cl)), activation)
        if dropout:
            cl.add_module(str(len(cl)), nn.Dropout(p=self.args.dropout))

        return cl

class SublayerConnection(customizedModule):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.init_mBloSA()

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        ret = x + self.dropout(sublayer(self.norm(x))[0])
        return ret,sublayer(self.norm(x))[1]

class SublayerConnection1(customizedModule):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection1, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)
        self.init_mBloSA()

    def init_mBloSA(self):
        self.g_W1 = self.customizedLinear(768, 768)
        self.g_W2 = self.customizedLinear(768, 768)
        self.g_b = nn.Parameter(torch.zeros(768))

        self.g_W1[0].bias.requires_grad = False
        self.g_W2[0].bias.requires_grad = False

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        ret = x + self.dropout(sublayer(self.norm(x)))
        return ret

# Each layer has two sub-layers. The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn      #多头注意力机制
        self.feed_forward = feed_forward    #前向神经网络
        # self.sublayer = clones(SublayerConnection(size, dropout), 2)

        self.sublayer = SublayerConnection(size, dropout)
        self.sublayer1 = SublayerConnection1(size, dropout)

        self.size = size

    def forward(self, x, mask,mask1,mask2):
        "Follow Figure 1 (left) for connections."
        x, att= self.sublayer(x, lambda x: self.self_attn(x, x, x, mask,mask1,mask2))
        # torch.Size([30, 10, 512])
        ret = self.sublayer1(x, self.feed_forward)
        return ret,att

class Encoder(nn.Module):
    "Core encoder is a stack of N layers"
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask,mask1,mask2):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x,att = layer(x, mask,mask1,mask2)
        return self.norm(x),att

class PromptSelector(nn.Module):
    def __init__(self, input_dim, pool_size,num_classes):
        super(PromptSelector, self).__init__()
        self.linear = nn.Linear(input_dim, pool_size)
        self.Encoder1 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, 768))

        self.classifier = nn.Linear(input_dim, num_classes)  # 新增的线性分类层

    def forward(self, text_features, prompt_pool):
        # Combine text and image features
        cls_tokens1 = repeat(self.cls_token1, '1 n d -> b n d', b=text_features.size(0))
        global_text_1 = torch.cat((cls_tokens1, text_features), dim=1)
        global_ouput_1, m11 = self.Encoder1(global_text_1, mask=None, mask1=None, mask2=None)

        # # Apply linear layer to obtain logits for prompt selection
        logits = self.linear(global_ouput_1[:,0,:])

        # Apply softmax to get probability distribution
        probabilities = F.softmax(logits, dim=1)   #4*10

        class_logits = self.classifier(global_ouput_1[:, 0, :])  # shape: (batch_size, num_classes)
        class_probabilities = F.softmax(class_logits, dim=1)  # shape: (batch_size, num_classes)
        entropy = -torch.sum(class_probabilities * torch.log(class_probabilities + 1e-9), dim=1)  # shape: (batch_size,)
        max_entropy = torch.log(torch.tensor(7, dtype=torch.float))  # 假设熵的最大可能值
        normalized_entropy = entropy / max_entropy

        adjusted_k = (1 + normalized_entropy * (prompt_pool.size(0) - 1)).long().clamp(1, prompt_pool.size(0)).tolist()
        # Get top-k indices based on probabilities
        selected_prompts = []
        max_k = max(adjusted_k)
        for i in range(probabilities.size(0)):
            top_values, top_indices = torch.topk(probabilities[i], adjusted_k[i], dim=0)
            selected_prompt = torch.index_select(prompt_pool, 0, top_indices)

            # Padding to max_k with zeros
            padding = torch.zeros(max_k - adjusted_k[i], prompt_pool.size(1)).to(selected_prompt.device)
            padded_prompt = torch.cat([selected_prompt, padding], dim=0)

            selected_prompts.append(padded_prompt)

        selected_prompts = torch.stack(selected_prompts)  # shape: (batch_size, max_k, prompt_dim)

        return selected_prompts,class_logits

class SectionOne(customizedModule):
    def __init__(self,pool_size,device):
        super(SectionOne, self).__init__()
        self.Encoder1 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, 768))
        self.selector = PromptSelector(768,pool_size, 7)

    def forward(self,section_feature, final_image, prompt_pool):
        selected_prompts,cls_sec = self.selector(section_feature, prompt_pool)
        cls_tokens1 = repeat(self.cls_token1, '1 n d -> b n d', b=section_feature.size(0))
        global_text_1 = torch.cat((cls_tokens1, section_feature, final_image, selected_prompts), dim=1)
        global_ouput_1, m11 = self.Encoder1(global_text_1, mask=None, mask1=None,mask2=None)

        return global_ouput_1,cls_sec

def create_window_attention_mask(interaction_range,device):
    batch_size, node_count = interaction_range.size()
    expanded_range = interaction_range.unsqueeze(-1).expand(batch_size, node_count, node_count)
    node_positions = torch.arange(node_count).unsqueeze(0).expand(batch_size, node_count, node_count).to(device)
    window_mask = (node_positions - node_positions.transpose(1, 2)).abs() <= expanded_range
    window_mask = window_mask.float()

    return window_mask

class SentenceFour(customizedModule):
    def __init__(self,pool_size,device):
        super(SentenceFour, self).__init__()
        self.device = device
        self.Encoder6 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.Encoder7 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.init_mBloSA()
        self.selector = PromptSelector(768, pool_size, 7)
        self.cls_token5 = nn.Parameter(torch.randn(1, 1, 768))

    def init_mBloSA(self):

        self.f_W3 = self.customizedLinear(768, 1)
        self.f_W4 = self.customizedLinear(768, 768)

    def forward(self, sentence_feature, final_image, prompt_pool):
        sentence_interaction_range = torch.round(1 + 2 * F.sigmoid(self.f_W3(self.f_W4(sentence_feature)).squeeze(-1)))
        section_mask_full = create_window_attention_mask(sentence_interaction_range,self.device)
        cls_tokens5 = repeat(self.cls_token5, '1 n d -> b n d', b=sentence_feature.size(0))
        global_text_41, att01 = self.Encoder6(sentence_feature, mask=section_mask_full.unsqueeze(1).expand(sentence_feature.size(0), 16, 100, 100), mask1=None, mask2=None)
        selected_prompts,cls_sec = self.selector(sentence_feature, prompt_pool)
        global_text_5 = torch.cat((cls_tokens5, global_text_41, final_image, selected_prompts), dim=1)
        global_text_51,att51 = self.Encoder7(global_text_5, mask=None,mask1 = None, mask2=None)

        return global_text_51,cls_sec

class Interaction(customizedModule):
    def __init__(self,device, dropout=None):
        super(Interaction, self).__init__()
        self.device = device
        self.Encoder1 = Encoder(EncoderLayer(768, MultiHeadedAttention(16, 768), PositionwiseFeedForward(768, 3072), 0.1), 1)
        self.cls_token1 = nn.Parameter(torch.randn(1, 1, 768))
    def forward(self, section_feature, sentence_feature, sec_sen_mask, sen_sen_mask):
        #生成我们需要的mask
        first_row = torch.ones(1,109).unsqueeze(0).unsqueeze(0).expand(section_feature.size(0), 16, 1, 109).to(self.device)
        second_row = torch.cat([torch.ones(8,1).unsqueeze(0).unsqueeze(0).expand(section_feature.size(0), 16, 8, 1).to(self.device),torch.eye(8).unsqueeze(0).unsqueeze(0).expand(section_feature.size(0), 16, 8, 8).to(self.device), sec_sen_mask.transpose(1,2).unsqueeze(1).expand(section_feature.size(0),16,8,100)],dim=3)
        third_row = torch.cat([torch.ones(100,1).unsqueeze(0).unsqueeze(0).expand(section_feature.size(0), 16, 100, 1).to(self.device),sec_sen_mask.unsqueeze(1).expand(section_feature.size(0),16,100,8),sen_sen_mask.unsqueeze(1).expand(section_feature.size(0), 16, 100, 100).to(self.device)], dim=3)
        fff_mask = torch.cat([first_row, second_row, third_row], dim=2)

        cls_tokens1 = repeat(self.cls_token1, '1 n d -> b n d', b=section_feature.size(0))
        global_text_1 = torch.cat((cls_tokens1, section_feature, sentence_feature), dim=1)
        global_text_11,atttt = self.Encoder1(global_text_1, mask=fff_mask, mask1=None,mask2=None)
        return global_text_11

class AAPDProcessor(BertProcessor):
    NAME = 'AAPD'
    NUM_CLASSES = 7
    IS_MULTILABEL = False

    def get_train_examples(self, data_dir):
        return self._create_examples(
                self._read_tsv(os.path.join(data_dir,'MMaterials', 'exMMaterials_train.tsv')), 'train')

    def get_dev_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'MMaterials', 'exMMaterials_dev.tsv')), 'dev')

    def get_test_examples(self, data_dir):
        return self._create_examples(
            self._read_tsv(os.path.join(data_dir, 'MMaterials', 'exMMaterials_test.tsv')), 'test')

    def _create_examples(self, lines, set_type):
        examples = []
        for (i, line) in enumerate(lines):

            guid = "%s-%s" % (set_type, i)
            text_a = line[1]
            label = line[0]
            examples.append(
                InputExample(guid=guid, text_a=text_a, text_b=None, label=label))
        return examples