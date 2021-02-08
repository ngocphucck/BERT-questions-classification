from torch import nn
from transformers import BertModel


class BertQuestionClassification(nn.Module):
    def __init__(self, n_classes, pretrained_name='bert-base-uncased'):
        super(BertQuestionClassification, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_name)
        self.dropout = nn.Dropout(0.1)
        self.classification = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        out = self.bert(input_ids, attention_mask)
        out = out.last_hidden_state[:, 0, :]
        out = self.dropout(out)
        out = self.classification(out)

        return out
