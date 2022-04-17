from pytools import memoize_method
import torch
import torch.nn.functional as F
import pytorch_pretrained_bert
from rankera import modeling_util
from transformers import BertModel, BertPreTrainedModel
from torch import nn



class BertForPairwiseLearning(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = 1 
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(p=0.05)
        self.classifier = nn.Linear(config.hidden_size, self.num_labels)
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        labels=None,
        side_input=None, task=1):

        if task==2:
            outputs_2 = self.bert(
                side_input['input_ids'],
                attention_mask=side_input['attention_mask'],
                token_type_ids=side_input['token_type_ids']
            )

            pooled_output_2 = outputs_2[0][:, 0, :]
            output = {'embedds':pooled_output_2}
            return output
        
        if task==1:
            outputs_1 = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids
            )
            pooled_output_1 = outputs_1[1]
            pooled_output_1 = self.dropout(pooled_output_1)
            logits_1 = self.classifier(pooled_output_1)

            output = {'logits': logits_1,}
            return output
