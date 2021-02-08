from torch.utils.data import Dataset
from transformers import BertTokenizer
import torch


tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)


class QuestionDataset(Dataset):
    def __init__(self, questions, labels, max_len=64):
        super(QuestionDataset, self).__init__()
        self.questions = questions
        self.labels = labels
        self.max_len = max_len
        pass

    def __getitem__(self, item):
        question = self.questions[item]
        label = self.labels[item]

        # Tokenize question: <CLS> - 101, <SEP> - 102, <PAD> - 0
        encoded_dict = tokenizer.encode_plus(
            question,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            return_attention_mask=True,
            return_tensors='pt'
        )

        input_ids = encoded_dict['input_ids']
        input_ids = input_ids.view(-1)
        attention_mask = encoded_dict['attention_mask']
        attention_mask = attention_mask.view(-1)

        return input_ids, attention_mask, torch.tensor(label)

    def __len__(self):
        return len(self.labels)


if __name__ == '__main__':
    dataset = QuestionDataset(["Who are you ?", "I'm phu"], [2, 4])
    print(dataset[0])
    print(dataset[1])
