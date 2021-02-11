from transformers import RobertaModel, RobertaTokenizer
import torch
import torch.nn as nn
import numpy as np

class bert_model(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = RobertaModel.from_pretrained('roberta-base')
        self.start = nn.Linear(768, 768)
        self.end = nn.Linear(768, 768)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        outputs = self.model(**inputs)
        last_hidden_states = outputs.last_hidden_state
        return self.softmax(self.start(last_hidden_states))

class bert_tokenizer():
    def __init__(self):
        self.tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    
    def get_token(self, text):
        return self.tokenizer(text, return_tensors="pt")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

model = bert_model().to(device)
tokenizer = bert_tokenizer()
model.train()
out = model(tokenizer.get_token('hello').to(device))
print(np.argmax(out.detach().cpu().numpy(), axis=2))