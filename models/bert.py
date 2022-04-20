import torch
from transformers import BertModel, BertPreTrainedModel
from torch import nn
from torch.functional import F

class BertClassifier(BertPreTrainedModel):

    def __init__(self, config, tokenizer):
        super(BertClassifier, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.classifier = nn.Linear(config.hidden_size, config.num_labels)
        self.loss_fn = nn.CrossEntropyLoss()
        self.tokenizer = tokenizer


        self.init_weights()

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        pooled_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

    def predict(self, text, device):
        # encoded_text = self.tokenizer(text, add_special_tokens=True, return_tensors='pt', padding=True)
        # logits = self.forward(**encoded_text)
        max_len = max([len(sentence) for sentence in text]) + 2
        max_len = min(max_len, 512)
        input_ids_list = torch.empty(len(text), max_len, device=device).long()
        token_type_ids_list = torch.empty(len(text), max_len, device=device).long()
        attention_mask_list = torch.empty(len(text), max_len, device=device).long()
        for i, tokens in enumerate(text):
            tokens = list(tokens)
            tokens = ['[CLS]'] + tokens[:max_len - 2] + ['[SEP]']
            seq_len = len(tokens)
            tokens = tokens + ['[PAD]'] * (max_len - seq_len)
            input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
            attention_mask = [1] * seq_len + [0] * (max_len - seq_len)
            token_type_ids = [0] * max_len

            assert len(input_ids) == max_len
            assert len(attention_mask) == max_len
            assert len(token_type_ids) == max_len

            input_ids_list[i, :] = torch.tensor(input_ids)
            token_type_ids_list[i, :] = torch.tensor(token_type_ids)
            attention_mask_list[i, :] = torch.tensor(attention_mask)

        logits = self.forward(input_ids_list, token_type_ids_list, attention_mask_list)
        return logits


