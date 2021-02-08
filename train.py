# 1.Import
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
import tqdm

from utils import get_data
from dataloader import QuestionDataset
from model import BertQuestionClassification

# 2.Dataloader
BATCH_SIZE = 40
questions, labels = get_data()

X_trainval, X_test, y_trainval, y_test = train_test_split(questions, labels, test_size=0.3, random_state=2021)
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.4, random_state=2021)

train_dataset = QuestionDataset(X_train, y_train)
val_dataset = QuestionDataset(X_val, y_val)
test_dataset = QuestionDataset(X_test, y_test)

train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
test_dataset = DataLoader(test_dataset, batch_size=len(X_test))

# 3.Train

# {'ABBREVIATION': 0, 'DESCRIPTION': 1, 'ENTITY': 2, 'HUMAN': 3, 'LOCATION': 4, 'NUMERIC': 5}

MODEL_SAVE_PATH = './data/model.pt'
model = BertQuestionClassification(n_classes=6)

N_EPOCHES = 20
lr = 2e-5
optimizer = Adam(model.parameters(), lr=lr)
loss_fn = CrossEntropyLoss()

train_losses = []
val_losses = []
val_f1scores = []

for epoch in range(N_EPOCHES):
    print("Epoch {}: ".format(epoch))

    train_batch_losses = []
    val_batch_losses = []
    val_batch_f1scores = []

    for (ids_train_batch, attention_mask_train_batch, y_train_batch) in tqdm.tqdm(train_dataloader):
        y_train_pred = model(ids_train_batch, attention_mask_train_batch)
        loss = loss_fn(y_train_pred, y_train_batch)
        train_batch_losses.append(loss.item())
        print("\nTrain batch loss: ", train_batch_losses[-1])
        print("Train batch f1-score: ", f1_score(torch.argmax(torch.nn.functional.softmax(y_train_pred, dim=-1), dim=-1),
              y_train_batch, average='macro'))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    for ids_val_batch, attention_mask_val_batch, y_val_batch in val_dataloader:
        y_val_pred = model(ids_val_batch, attention_mask_val_batch)
        loss = loss_fn(y_val_pred, y_val_batch)
        val_batch_losses.append(loss.item())
        val_batch_f1scores.append(
            f1_score(torch.argmax(torch.nn.functional.softmax(y_val_pred, dim=-1), dim=-1),
                     y_val_batch, average='macro')
        )
        print("Validation batch f1-score: ", val_batch_f1scores[-1])

    train_losses.append(sum(train_batch_losses) / len(train_batch_losses))
    val_losses.append(sum(val_batch_losses) / len(val_batch_losses))
    val_f1scores.append(sum(val_batch_f1scores) / len(val_batch_f1scores))
    torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print("Train loss: ", train_losses[-1])
    print("Validation loss: ", val_losses[-1])
    print("Validation F1 score: ", val_f1scores[-1])
