
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch.nn.functional as F

from process_data import Dataset, DataModule,prepare_data
LABEL_Columns = ["question", "entity","answer","start_idx","end_idx","sentence_length"]
dev = 'cuda'

class EntityPredictor(pl.LightningModule):
    def __init__(self, n_classes: int, entity_embeddings, candidate_embedding, entity_dict, steps_per_epoch=None,
                 n_epochs=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)

        # BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)

        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, self.bert.config.num_labels)  # 768 to 32
        self.cls_output = nn.Linear(self.bert.config.hidden_size, 3)
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.criterion = nn.CrossEntropyLoss(ignore_index=64)
        self.total_acc = 0
        self.total_acc_cls = 0
        self.train_counter = 0
        self.total_loss = 0

        self.val_counter = 0
        self.total_val_loss = 0

        self.total_val_acc = 0

        # TODO ans pred....
        self.entity_embeddings = entity_embeddings
        self.entity_dict = entity_dict
        # TODO needs reset for each step
        self.head_coord_list = torch.tensor([[0, 0]]).to(dev)
        self.head_words = []
        self.head_embedding_list = torch.tensor([]).to(dev)
        # TODO question related layers
        self.relation_dim = 200 * 2
        self.hidden2rel = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),  # TODO 768 , 512
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, self.relation_dim),
            nn.ReLU()
        )
        self.reshape_bef = torch.nn.Linear(768, 1600).to(dev)
        self.reshape_pos = torch.nn.Linear(1600, 400).to(dev)
        self.hidden2rel_conv = nn.Sequential(
            torch.nn.Conv2d(1, 32, 3, 1, 1).to(dev),
            nn.ReLU().to(dev),
            nn.Dropout(0.1).to(dev),
            nn.BatchNorm2d(32).to(dev),
            torch.nn.Conv2d(32, 64, 3, 1, 1).to(dev),
            nn.ReLU().to(dev),
            nn.Dropout(0.1).to(dev),
            nn.BatchNorm2d(64).to(dev),
            torch.nn.Conv2d(64, 64, 3, 1, 1).to(dev),
            nn.ReLU().to(dev),
            nn.Dropout(0.1).to(dev),
            nn.BatchNorm2d(64).to(dev),
            torch.nn.Conv2d(64, 64, 3, 1, 1).to(dev),
            nn.ReLU().to(dev),
            nn.Dropout(0.1).to(dev),
            nn.BatchNorm2d(64).to(dev),
            torch.nn.Conv2d(64, 32, 3, 1, 1).to(dev),
            nn.ReLU().to(dev),
            nn.Dropout(0.1).to(dev),
            nn.BatchNorm2d(32).to(dev),
            torch.nn.Conv2d(32, 1, 3, 1, 1).to(dev),
            nn.ReLU().to(dev),
            nn.Dropout(0.1).to(dev),
            nn.BatchNorm2d(1).to(dev),

        )

        self.candidate_embedding = candidate_embedding
        self.loss = self.kge_loss
        self._klloss = torch.nn.KLDivLoss(reduction='sum')
        self.bce = torch.nn.BCEWithLogitsLoss()

        # TODO: method 1
        # self.linear = nn.Linear(800, 43234).to(dev)


    def kge_loss(self, scores, targets):
        # loss = torch.mean(scores*targets)
        return self._klloss(
            F.log_softmax(scores, dim=1), F.normalize(targets.float(), p=1, dim=1)
        )

    def forward(self, input_ids, attention_mask, start_idx_label, end_idx_label, cls_label, ans):
        output = self.bert(input_ids, attention_mask)

        sequence_output = output[0]
        pooled_outputs = output[1]
        logits = self.qa_outputs(sequence_output)
        cls_output = self.cls_output(pooled_outputs)

        # print(sequence_output.shape, logits.shape)  # 128 is max token length
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        loss = 0

        if start_idx_label is not None and end_idx_label is not None:
            ignore_idx = start_logits.size(1)
            # print(ignore_idx)

            # start_logits.clamp_(0,ignore_idx)
            # end_logits.clamp_(0,ignore_idx)

            # print(start_logits.shape, end_logits.shape)
            start_loss = self.criterion(start_logits, start_idx_label)
            end_loss = self.criterion(end_logits, end_idx_label)
            cls_loss = self.criterion(cls_output, cls_label)
            loss = (start_loss + end_loss + cls_loss) / 3

        return loss, start_logits, end_logits, cls_output, pooled_outputs, sequence_output

    def training_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_idx_label = batch["start_idx_label"]
        end_idx_label = batch["end_idx_label"]
        cls_label = batch["hop_type"]
        ans_label = batch['answer']  # todo batch x 43234

        loss, start_logits, end_logits, cls_output, pooled_outputs, sequence_output = self(input_ids, attention_mask,
                                                                                           start_idx_label,
                                                                                           end_idx_label, cls_label,
                                                                                           ans_label)

        label = torch.cat((start_idx_label.unsqueeze(0), end_idx_label.unsqueeze(0)), 0)
        label = torch.transpose(label, 0, 1)

        _, start_idx = torch.topk(start_logits, k=3, dim=1)
        _, end_idx = torch.topk(end_logits, k=3, dim=1)
        pred_idx = torch.cat((start_idx.unsqueeze(0), end_idx.unsqueeze(0)), 0)
        pred_idx = torch.transpose(pred_idx, 0, 1)
        pred_idx = torch.transpose(pred_idx, 1, 2)
        correct = 0
        counter2 = 0
        for group_pred in pred_idx[:]:
            # print(group_pred)
            higest = group_pred[0]
            combines = None
            sub_head = higest[0]
            sub_tail = higest[1]
            # print(higest,sub_head,sub_tail)
            second_higest = group_pred[1]
            third_higest = group_pred[2]
            if sub_head < second_higest[1]:
                sub_ = torch.cat((sub_head.unsqueeze(0), second_higest[1].unsqueeze(0)), 0)
                combines = torch.cat((higest.unsqueeze(0), sub_.unsqueeze(0)), 0)
            if sub_head < third_higest[1]:
                if combines is None:
                    combines = higest.unsqueeze(0)
                sub_ = torch.cat((sub_head.unsqueeze(0), third_higest[1].unsqueeze(0)), 0)
                combines = torch.cat((combines, sub_.unsqueeze(0)), 0)
            if second_higest[0] < sub_tail:
                if combines is None:
                    combines = higest.unsqueeze(0)
                sub_ = torch.cat((second_higest[0].unsqueeze(0), sub_tail.unsqueeze(0)), 0)
                combines = torch.cat((combines, sub_.unsqueeze(0)), 0)
            if third_higest[0] < sub_tail:
                if combines is None:
                    combines = higest.unsqueeze(0)
                sub_ = torch.cat((third_higest[0].unsqueeze(0), sub_tail.unsqueeze(0)), 0)
                combines = torch.cat((combines, sub_.unsqueeze(0)), 0)
            # print(combines)
            # print(1 + '1')
            if combines is None:
                combines = higest.unsqueeze(0)
            combines = torch.cat((combines, second_higest.unsqueeze(0)), 0)

            flag = False
            for each in combines:
                if torch.all(torch.eq(each, label[counter2]), dim=0):
                    correct += 1
                    flag = True
                    self.head_coord_list = torch.cat((self.head_coord_list, each.unsqueeze(0)),
                                                     0)  # predicted correct head coord
                    break
            if not flag:
                self.head_coord_list = torch.cat((self.head_coord_list, label[counter2].unsqueeze(0)),
                                                 0)  # labeled correct head coord
            counter2 += 1
        # correct = torch.all(torch.eq(pred, label), dim=1).sum()
        acc = correct / BATCH_SIZE

        acc_cls = (torch.argmax(cls_output, dim=1) == cls_label).sum() / BATCH_SIZE
        # print(1 + '1')
        self.log("training loss", loss, prog_bar=True, logger=True)
        self.log("acc", acc, prog_bar=True, logger=True)
        self.log("cls_acc", acc_cls, prog_bar=True, logger=True)

        # todo 结合部分开始！！！！！
        # TODO get head embeddings
        self.head_coord_list = self.head_coord_list[1:]
        # print(self.head_coord_list.shape)
        questions = batch['question']
        counter3 = 0
        for each_question in questions:
            question_tokens = each_question.split(" ")
            a_coord = list(self.head_coord_list[counter3].detach().cpu().numpy())
            head_entity = question_tokens[a_coord[0]:a_coord[1] + 1]
            head_entity = " ".join(head_entity).strip(",")
            try:
                this_e = torch.FloatTensor(entity_embeddings[self.entity_dict[head_entity]]).to(dev)
            except:
                print(question_tokens)
            # print(this_e.shape)
            self.head_embedding_list = torch.cat((self.head_embedding_list, this_e.unsqueeze(0)), 0)
            self.head_words.append(head_entity)
            counter3 += 1

        # print(self.head_words)
        # print(self.head_embedding_list.shape) #TODO batch_size * 400
        # TODO question embedding
        '''
        states = sequence_output.transpose(1, 0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        '''
        question_embedding = pooled_outputs  # TODO batch_size * 768
        # rel_embedding = self.hidden2rel(question_embedding)  # TODO batch_size * 768

        # TODO conv_
        question_embedding = self.reshape_bef(question_embedding)
        question_embedding = question_embedding.view(-1, 1, 40, 40)
        rel_embedding = self.hidden2rel_conv(question_embedding)  # TODO batch_size * 768
        rel_embedding = rel_embedding.view(rel_embedding.shape[0], -1)
        rel_embedding = self.reshape_pos(rel_embedding)
        # print(rel_embedding.shape)

        # print(self.head_words)

        pred = complex.score(self.head_embedding_list, rel_embedding, self.candidate_embedding)

        # ans_label = ((1.0 - 0.5) * ans_label) + (1.0 / ans_label.size(1))
        loss_ans = self.bce(pred, ans_label)  # self.kge_loss(pred, ans_label)

        '''
        print(questions[20])

        b,d = kg.get_candidates_embeddings(k=1, head=self.head_words[20])
        c = nn.Embedding.from_pretrained(torch.FloatTensor(b), freeze=True).to('cuda')
        a = complex.score(self.head_embedding_list[20].unsqueeze(0), rel_embedding[20].unsqueeze(0),   c  )

        top_results = complex.get_top_k(a,1)
        for _, idx in zip(top_results[0], top_results[1]):
            print(d[idx.item()])

        #print(1+'1')
        '''

        # TODO reset

        self.head_coord_list = torch.tensor([[0, 0]]).to(dev)
        self.head_words = []
        self.head_embedding_list = torch.tensor([]).to(dev)

        self.total_loss += loss_ans.item()
        self.train_counter += 1
        return loss_ans  # {"loss": loss}

    def training_epoch_end(self, validation_step_outputs):
        self.log("total_loss", self.total_loss / self.train_counter, prog_bar=True, logger=True)
        print(self.total_loss / self.train_counter)
        self.total_loss = 0
        self.train_counter = 0

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_idx_label = batch["start_idx_label"]
        end_idx_label = batch["end_idx_label"]
        cls_label = batch["hop_type"]
        ans_label = batch['answer']  # todo batch x 43234

        loss, start_logits, end_logits, cls_output, pooled_outputs, sequence_output = self(input_ids, attention_mask,
                                                                                           start_idx_label,
                                                                                           end_idx_label, cls_label,
                                                                                           ans_label)

        label = torch.cat((start_idx_label.unsqueeze(0), end_idx_label.unsqueeze(0)), 0)
        label = torch.transpose(label, 0, 1)

        _, start_idx = torch.topk(start_logits, k=3, dim=1)
        _, end_idx = torch.topk(end_logits, k=3, dim=1)
        pred_idx = torch.cat((start_idx.unsqueeze(0), end_idx.unsqueeze(0)), 0)
        pred_idx = torch.transpose(pred_idx, 0, 1)
        pred_idx = torch.transpose(pred_idx, 1, 2)
        correct = 0
        counter2 = 0
        for group_pred in pred_idx[:]:
            # print(group_pred)
            higest = group_pred[0]
            combines = None
            sub_head = higest[0]
            sub_tail = higest[1]
            # print(higest,sub_head,sub_tail)
            second_higest = group_pred[1]
            third_higest = group_pred[2]
            if sub_head < second_higest[1]:
                sub_ = torch.cat((sub_head.unsqueeze(0), second_higest[1].unsqueeze(0)), 0)
                combines = torch.cat((higest.unsqueeze(0), sub_.unsqueeze(0)), 0)
            if sub_head < third_higest[1]:
                if combines is None:
                    combines = higest.unsqueeze(0)
                sub_ = torch.cat((sub_head.unsqueeze(0), third_higest[1].unsqueeze(0)), 0)
                combines = torch.cat((combines, sub_.unsqueeze(0)), 0)
            if second_higest[0] < sub_tail:
                if combines is None:
                    combines = higest.unsqueeze(0)
                sub_ = torch.cat((second_higest[0].unsqueeze(0), sub_tail.unsqueeze(0)), 0)
                combines = torch.cat((combines, sub_.unsqueeze(0)), 0)
            if third_higest[0] < sub_tail:
                if combines is None:
                    combines = higest.unsqueeze(0)
                sub_ = torch.cat((third_higest[0].unsqueeze(0), sub_tail.unsqueeze(0)), 0)
                combines = torch.cat((combines, sub_.unsqueeze(0)), 0)
            # print(combines)
            # print(1 + '1')
            if combines is None:
                combines = higest.unsqueeze(0)
            combines = torch.cat((combines, second_higest.unsqueeze(0)), 0)

            flag = False
            for each in combines:
                if torch.all(torch.eq(each, label[counter2]), dim=0):
                    correct += 1
                    flag = True
                    self.head_coord_list = torch.cat((self.head_coord_list, each.unsqueeze(0)),
                                                     0)  # predicted correct head coord
                    break
            if not flag:
                self.head_coord_list = torch.cat((self.head_coord_list, label[counter2].unsqueeze(0)),
                                                 0)  # labeled correct head coord
            counter2 += 1
        # correct = torch.all(torch.eq(pred, label), dim=1).sum()
        acc = correct / BATCH_SIZE

        acc_cls = (torch.argmax(cls_output, dim=1) == cls_label).sum() / BATCH_SIZE
        # print(1 + '1')
        self.log("training loss", loss, prog_bar=True, logger=True)
        self.log("acc", acc, prog_bar=True, logger=True)
        self.log("cls_acc", acc_cls, prog_bar=True, logger=True)

        # todo 结合部分开始！！！！！
        # TODO get head embeddings
        self.head_coord_list = self.head_coord_list[1:]
        # print(self.head_coord_list.shape)
        questions = batch['question']
        counter3 = 0
        for each_question in questions:
            question_tokens = each_question.split(" ")
            a_coord = list(self.head_coord_list[counter3].detach().cpu().numpy())
            head_entity = question_tokens[a_coord[0]:a_coord[1] + 1]
            head_entity = " ".join(head_entity).strip(",")
            try:
                this_e = torch.FloatTensor(entity_embeddings[self.entity_dict[head_entity]]).to(dev)
            except:
                print(question_tokens)
            # print(this_e.shape)
            self.head_embedding_list = torch.cat((self.head_embedding_list, this_e.unsqueeze(0)), 0)
            self.head_words.append(head_entity)
            counter3 += 1

        # print(self.head_words)
        # print(self.head_embedding_list.shape) #TODO batch_size * 400
        # TODO question embedding
        '''
        states = sequence_output.transpose(1, 0)
        cls_embedding = states[0]
        question_embedding = cls_embedding
        '''
        question_embedding = pooled_outputs  # TODO batch_size * 768

        # rel_embedding = self.hidden2rel(question_embedding)  # TODO batch_size * 768
        # print(self.head_words)
        # TODO conv_
        question_embedding = self.reshape_bef(question_embedding)
        question_embedding = question_embedding.view(-1, 1, 40, 40)
        rel_embedding = self.hidden2rel_conv(question_embedding)  # TODO batch_size * 768
        rel_embedding = rel_embedding.view(rel_embedding.shape[0], -1)
        rel_embedding = self.reshape_pos(rel_embedding)

        k = 1 # 1hop for now
        pred = complex.score(self.head_embedding_list, rel_embedding, self.candidate_embedding)
        neighbor_one_hot = torch.tensor(kg.get_neighbour_onehot(heads=self.head_words, ks=[k]*len(self.head_words),smoothing=SMOOTH)).to(dev)
        pred = torch.mul(pred, neighbor_one_hot)
        # ans_label = ((1.0 - 0.5) * ans_label) + (1.0 / ans_label.size(1))
        loss_ans = self.bce(pred, ans_label)  # self.kge_loss(pred, ans_label)

        # calculate acc
        local_count = 0
        acc = 0
        for i in torch.argmax(pred, 1):
            if ans_label[local_count][i] == 1:
                acc += 1
            local_count += 1
        acc = acc / local_count
        self.total_val_acc += acc

        '''
        print(questions[0])

        b,d = kg.get_candidates_embeddings(k=1, head=self.head_words[0])
        c = nn.Embedding.from_pretrained(torch.FloatTensor(b), freeze=True).to('cuda')
        a = complex.score(self.head_embedding_list[0].unsqueeze(0), rel_embedding[0].unsqueeze(0),   c  )

        top_results = complex.get_top_k(a,1)
        for _, idx in zip(top_results[0], top_results[1]):
            print(d[idx.item()])
        '''

        # TODO reset

        self.head_coord_list = torch.tensor([[0, 0]]).to(dev)
        self.head_words = []
        self.head_embedding_list = torch.tensor([]).to(dev)

        self.total_val_loss += loss_ans.item()
        self.val_counter += 1
        return loss_ans  # {"loss": loss}

    def validation_epoch_end(self, validation_step_outputs):
        self.log("total_val_loss", self.total_val_loss / self.val_counter, prog_bar=True, logger=True)
        print(self.total_val_loss / self.val_counter)
        print("total_val_acc", self.total_val_acc / self.val_counter)
        self.total_val_loss = 0
        self.val_counter = 0
        self.total_val_acc = 0

    def test_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_idx_label = batch["start_idx_label"]
        end_idx_label = batch["end_idx_label"]
        cls_label = batch["hop_type"]
        loss, start_logits, end_logits, cls_output = self(input_ids, attention_mask, start_idx_label, end_idx_label,
                                                          cls_label)

        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.00001)


df = prepare_data()
print(df.shape)
val_df, train_df = train_test_split(df, test_size=0.9)
print(train_df.shape, val_df.shape)
# BERT_MODEL_NAME = "bert-base-cased"
BERT_MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"
# tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

print(tokenizer)
"""load embeddings"""
from all_kg import KG, Complex
kg = KG()
complex = Complex('cuda')
entity_embeddings, relation_embeddings, entity_dict, relation_dict = kg.load_kg_embeddings()

# TODO 先获取所有的candidtae embeddings 训练时候用， 一直保持不变
pretrained_embeddings, candidate_dict = kg.get_candidates_embeddings(
    k=0)  # when k=0 find all entities in KG, else find head k neighbours
candidate_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=True).to('cuda')
print("entity_embeddings", entity_embeddings.shape)
print("relation_embeddings", relation_embeddings.shape)

N_EPOCHS = 500
BATCH_SIZE = 32
SMOOTH = 0.0
data_module = DataModule(entity_dict, train_df[:], val_df[:], tokenizer, BATCH_SIZE)
data_module.setup()

model = EntityPredictor(
    n_classes=2,
    entity_embeddings=entity_embeddings,
    candidate_embedding=candidate_embedding,
    entity_dict=entity_dict,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    n_epochs=N_EPOCHS

)
trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=1, progress_bar_refresh_rate=10, checkpoint_callback=False)
trainer.fit(model, data_module)


