import warnings
import sys
sys.path.append('./datasets')
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from sklearn import metrics
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset
from tqdm import tqdm
import os
from torch.autograd import Variable
from bert_processors.abstract_processor import convert_examples_to_features,convert_image_to_features
from utils.preprocessing import pad_input_matrix
import pickle
# Suppress warnings from sklearn.metrics
warnings.filterwarnings('ignore')
import torch

class BertEvaluator(object):
    def __init__(self, model, processor, tokenizer, args, split='dev'):
        self.args = args
        self.model = model
        self.processor = processor
        self.tokenizer = tokenizer
        if split == 'test':
            self.eval_examples = self.processor.get_test_examples(args.data_dir)
            self.eval_image_1 = convert_image_to_features(os.path.join(self.args.data_dir,'MMaterials', 'test_vgg_clip.img'), self.args.image_max_len)
        else:
            self.eval_examples = self.processor.get_dev_examples(args.data_dir)
            self.eval_image_1 = convert_image_to_features(os.path.join(self.args.data_dir,'MMaterials', 'dev_vgg_clip.img'), self.args.image_max_len)

    def get_scores(self, silent=False):
        eval_features = convert_examples_to_features(
            self.eval_examples, self.args.max_seq_length, self.tokenizer)

        unpadded_input_ids = [f.input_ids for f in eval_features]
        unpadded_input_mask = [f.input_mask for f in eval_features]
        unpadded_segment_ids = [f.segment_ids for f in eval_features]
        unpadded_sentence_mask = [f.sentence_mask for f in eval_features]
        padded_input_ids = torch.tensor(unpadded_input_ids, dtype=torch.long)
        padded_input_mask = torch.tensor(unpadded_input_mask, dtype=torch.long)
        padded_segment_ids = torch.tensor(unpadded_segment_ids, dtype=torch.long)
        padded_sentence_mask = torch.tensor(unpadded_sentence_mask, dtype=torch.long)
        padded_image_ids = torch.tensor(self.eval_image_1, dtype=torch.float)
        label_ids = torch.tensor([f.label_id for f in eval_features], dtype=torch.long)

        eval_data = TensorDataset(padded_input_ids, padded_input_mask, padded_segment_ids,padded_sentence_mask, label_ids,padded_image_ids)
        eval_sampler = SequentialSampler(eval_data)
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=self.args.eval_batch_size,drop_last = True)

        self.model.eval()

        total_loss = 0
        nb_eval_steps, nb_eval_examples = 0, 0
        predicted_labels, target_labels = list(), list()

        for input_ids, input_mask, segment_ids,sentence_mask,label_ids, eval_image_1 in tqdm(eval_dataloader, desc="Evaluating", disable=silent):
            input_ids = input_ids.to(self.args.device)
            input_mask = input_mask.to(self.args.device)
            segment_ids = segment_ids.to(self.args.device)
            sentence_mask = sentence_mask.to(self.args.device)
            label_ids = label_ids.to(self.args.device)
            eval_image_1 =  eval_image_1.to(self.args.device)
            with torch.no_grad():
                logits,cls_loss_sec1, cls_loss_sen1,cls_loss_sec2, cls_loss_sen2= self.model(input_ids.view(-1, self.args.max_seq_length),eval_image_1,sentence_mask, segment_ids.view(-1, self.args.max_seq_length), input_mask.view(-1, self.args.max_seq_length))

            if self.args.is_multilabel:
                predicted_labels.extend(F.sigmoid(logits).round().long().cpu().detach().numpy())
                target_labels.extend(label_ids.cpu().detach().numpy())
                loss = F.binary_cross_entropy_with_logits(logits, label_ids.float(), size_average=False)
            else:
                predicted_labels.extend(torch.argmax(logits, dim=1).cpu().detach().numpy())
                target_labels.extend(torch.argmax(label_ids, dim=1).cpu().detach().numpy())
                loss = F.cross_entropy(logits, torch.argmax(label_ids, dim=1))

            if self.args.n_gpu > 1:
                loss = loss.mean()
            if self.args.gradient_accumulation_steps > 1:
                loss = loss / self.args.gradient_accumulation_steps
            total_loss += loss.item()

            nb_eval_examples += input_ids.size(0)
            nb_eval_steps += 1

        predicted_labels, target_labels = np.array(predicted_labels), np.array(target_labels)
        accuracy = metrics.accuracy_score(target_labels, predicted_labels)
        precision = metrics.precision_score(target_labels, predicted_labels, average='macro')
        recall = metrics.recall_score(target_labels, predicted_labels, average='macro')
        f1 = metrics.f1_score(target_labels, predicted_labels, average='macro')
        avg_loss = total_loss / nb_eval_steps

        return [accuracy, precision, recall, f1, avg_loss], ['accuracy', 'precision', 'recall', 'f1', 'avg_loss']
