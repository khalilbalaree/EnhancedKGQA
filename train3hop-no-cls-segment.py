
from networkx.algorithms import matching
import torch
import torch.nn as nn
from torch.nn.modules.activation import ReLU, Threshold
from transformers import BertTokenizer, BertModel, AutoTokenizer, AutoModel
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
import torch.nn.functional as F
import itertools
import numpy as np

from process_data import Dataset, DataModule,prepare_data
LABEL_Columns = ["question", "entity","answer","start_idx","end_idx","sentence_length"]
dev = 'cuda'

class EntityPredictor(pl.LightningModule):
    def __init__(self, n_classes: int, relation_embedding, candidate_embedding, entity_dict, steps_per_epoch=None,
                 n_epochs=None):
        super().__init__()
        self.bert = AutoModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)
        # BertModel.from_pretrained(BERT_MODEL_NAME, return_dict=True)

        self.qa_outputs = nn.Linear(self.bert.config.hidden_size, self.bert.config.num_labels)  # 768 to 32
        self.n_epochs = n_epochs
        self.steps_per_epoch = steps_per_epoch
        self.criterion = nn.CrossEntropyLoss(ignore_index=32)
        self.total_acc = 0
        self.total_acc_cls = 0
        self.train_counter = 0
        self.total_loss = 0
        self.cls_acc = 0

        self.val_counter = 0
        self.total_val_loss = 0

        self.total_val_acc = 0
        self.total_val_acc1 = 0

        # TODO ans pred....
        self.relation_embedding = relation_embedding # used for relation matching
        self.entity_dict = entity_dict
        # TODO needs reset for each step
        self.head_coord_list = torch.tensor([[0, 0]]).to(dev)
        self.head_words = []
        self.head_embedding_list = torch.tensor([]).to(dev)
        # TODO question related layers
        self.relation_dim = 200 * 2

        self.conv1d_shape_400 = torch.nn.Linear(740, 400).to(dev)
        self.test_conv = nn.Sequential(
            torch.nn.Conv1d(1, 128, kernel_size=5,stride=1).to(dev),
            nn.ReLU().to(dev),
            nn.BatchNorm1d(128).to(dev),
            torch.nn.Conv1d(128, 256, kernel_size=5,stride=1).to(dev),
            nn.ReLU().to(dev),
            nn.BatchNorm1d(256).to(dev),
            torch.nn.Conv1d(256, 512, kernel_size=5,stride=1).to(dev),
            nn.ReLU().to(dev),
            nn.BatchNorm1d(512).to(dev),
            torch.nn.Conv1d(512, 512, kernel_size=5, stride=1).to(dev),
            nn.ReLU().to(dev),
            nn.BatchNorm1d(512).to(dev),
            torch.nn.Conv1d(512, 256, kernel_size=5,stride=1).to(dev),
            nn.ReLU().to(dev),
            nn.BatchNorm1d(256).to(dev),
            torch.nn.Conv1d(256, 128, kernel_size=5,stride=1).to(dev),
            nn.ReLU().to(dev),
            nn.BatchNorm1d(128).to(dev),
            torch.nn.Conv1d(128, 1, kernel_size=5,stride=1).to(dev),
            nn.ReLU().to(dev),
            nn.BatchNorm1d(1).to(dev),
        )

        

        self.candidate_embedding = candidate_embedding
        self.loss = self.kge_loss
        self._klloss = torch.nn.KLDivLoss(reduction='sum')
        self.bce = torch.nn.BCEWithLogitsLoss()
        self.sigmoid = nn.Sigmoid()

        self.hop = 3

        self.cls_output = nn.Sequential(
            nn.Linear(768, 18)
        )

        #nn.Linear(self.bert.config.hidden_size, 18)




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
            # cls_loss = self.criterion(cls_output, cls_label)
            # loss = (start_loss + end_loss + cls_loss) / 3
            loss = (start_loss + end_loss) / 2

        return loss, start_logits, end_logits, cls_output, pooled_outputs

    def training_step(self, batch, batch_idx, optimizer_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_idx_label = batch["start_idx_label"]
        end_idx_label = batch["end_idx_label"]
        cls_label = batch["hop_type"]
        ans_label = batch['answer']  # todo batch x 43234
        this_hop_type = batch['this_hop_type']

        loss, start_logits, end_logits, cls_output, pooled_outputs = self(input_ids, attention_mask,
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

        # acc_cls = (torch.argmax(cls_output, dim=1) == cls_label).sum() / BATCH_SIZE
        # print(1 + '1')
        self.log("training loss", loss, prog_bar=True, logger=True)
        self.log("acc", acc, prog_bar=True, logger=True)
        # self.log("cls_acc", acc_cls, prog_bar=True, logger=True)


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

        question_embedding = pooled_outputs  # TODO batch_size * 768
        # rel_embedding = self.hidden2rel(question_embedding)  # TODO batch_size * 768

        # TODO conv_
        question_embedding = question_embedding.unsqueeze(1)
        relation_embedding = self.test_conv(question_embedding)

        rel_embedding = self.conv1d_shape_400(relation_embedding.squeeze(1))
        # print(rel_embedding.shape)
        # rel_embedding = self.apply_nonlinar(question_embedding)

        # print(self.head_words)
        # print(questions)

        #bestscore, ind = torch.topk(scores_sigmoid,k=3, dim=1)
        #print(this_hop_type)
        cls_label = torch.tensor(kg.path_relations_batch(self.head_words, ans_label, this_hop_type)).to(dev)

        #print(cls_output[0], torch.where(cls_label[0]>0))
        weight = cls_label *5
        cls_loss = F.binary_cross_entropy_with_logits(cls_output, cls_label,pos_weight=weight)
        self.log("cls_loss", cls_loss, prog_bar=True, logger=True)
        pred = complex.score(self.head_embedding_list, rel_embedding, self.candidate_embedding)




        label_length = 20
        new_label = torch.zeros(label_length).unsqueeze(0).to(dev)
        new_pred = torch.zeros(label_length).unsqueeze(0).to(dev)
        best_labels = torch.zeros(label_length).unsqueeze(0).to(dev)
        best_labels_pred_idx = torch.zeros(label_length).unsqueeze(0).to(dev)
        for i in range(pred.size(0)):
            each_batch = pred[i]
            this_ans = ans_label[i]
            values, indices = torch.topk(each_batch, label_length)
            new_label = torch.cat((new_label, this_ans[indices].unsqueeze(0)), dim=0)
            new_pred = torch.cat((new_pred, each_batch[indices].unsqueeze(0)), dim=0)
            # todo another loss

            ind = torch.where(this_ans > 0)[0]
            if ind.shape[0] > label_length:
                a = torch.randperm(ind.shape[0])[:label_length]
                ind = ind[a]  # todo pick up first 25 gt labels' positions
                gt_labels = this_ans[ind]
                crossponding_pred_pos = each_batch[ind]
            else:
                gt_labels = this_ans[ind]
                crossponding_pred_pos = each_batch[ind]
                diff = label_length - gt_labels.shape[0]
                v, ind = torch.topk(each_batch, diff)
                gt_labels_remains = this_ans[ind]
                crossponding_pred_pos_remains = each_batch[ind]
                composite_gt_labels = torch.cat((gt_labels, gt_labels_remains))
                composite_crossponding_pred_pos = torch.cat((crossponding_pred_pos, crossponding_pred_pos_remains))
                # print(gt_labels,gt_labels_remains, composite_gt_labels)
                gt_labels = composite_gt_labels
                crossponding_pred_pos = composite_crossponding_pred_pos
            best_labels = torch.cat((best_labels, gt_labels.unsqueeze(0)), dim=0)
            best_labels_pred_idx = torch.cat((best_labels_pred_idx, crossponding_pred_pos.unsqueeze(0)), dim=0)
            # print(best_labels)

            # print(gt_labels,gt_labels.shape)
            # print(ind)
        new_label = new_label[1:]
        new_pred = new_pred[1:]
        best_labels = best_labels[1:]
        best_labels_pred_idx = best_labels_pred_idx[1:]
        #loss_pred = self._klloss(F.log_softmax(new_pred, dim=1), F.normalize(new_label.float(), p=1, dim=1))
        w1 = new_label * 5
        loss_pred = F.binary_cross_entropy_with_logits(new_pred,new_label,pos_weight=w1)
        w3 = best_labels * 1
        loss_label = F.binary_cross_entropy_with_logits(best_labels_pred_idx,best_labels,pos_weight=w3)
        '''
        loss_ans = F.binary_cross_entropy_with_logits(pred,ans_label,pos_weight=ans_label * 100)#loss_pred + loss_label
        self.log("loss_ans", loss_ans, prog_bar=True, logger=True)
        '''
        loss_ans = loss_pred + loss_label
        self.log("loss_pred", loss_pred, prog_bar=True, logger=True)
        self.log("loss_label", loss_label, prog_bar=True, logger=True)



        #loss_ans = self.bce(pred, ans_label)  # self.kge_loss(pred, ans_label)

        # weight = ((1.0-0.1)*ans_label) + (0.1 / ans_label.size(1))
        # loss_ans = F.binary_cross_entropy_with_logits(pred, ans_label, pos_weight=weight)


        # TODO reset

        self.head_coord_list = torch.tensor([[0, 0]]).to(dev)
        self.head_words = []
        self.head_embedding_list = torch.tensor([]).to(dev)

        self.total_loss += loss_ans.item()
        self.train_counter += 1
        if optimizer_idx == 0:
            return loss_ans
        if optimizer_idx == 1:
            return cls_loss

    def training_epoch_end(self, validation_step_outputs):
        self.log("total_loss", self.total_loss / self.train_counter, prog_bar=True, logger=True)
        #print(self.total_loss / self.train_counter)
        self.total_loss = 0
        self.train_counter = 0

    def validation_step(self, batch, batch_idx):
        input_ids = batch["input_ids"]
        attention_mask = batch["attention_mask"]
        start_idx_label = batch["start_idx_label"]
        end_idx_label = batch["end_idx_label"]
        cls_label = batch["hop_type"]
        ans_label = batch['answer']  # todo batch x 43234
        this_hop_type = batch['this_hop_type']

        loss, start_logits, end_logits, cls_output, pooled_outputs = self(input_ids, attention_mask,
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

        # acc_cls = (torch.argmax(cls_output, dim=1) == cls_label).sum() / BATCH_SIZE
        # print(1 + '1')
        self.log("training loss", loss, prog_bar=True, logger=True)
        self.log("acc", acc, prog_bar=True, logger=True)
        # self.log("cls_acc", acc_cls, prog_bar=True, logger=True)

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
        question_embedding = question_embedding.unsqueeze(1)
        relation_embedding = self.test_conv(question_embedding)
        rel_embedding = self.conv1d_shape_400(relation_embedding.squeeze(1))


        
        pred = complex.score(self.head_embedding_list, rel_embedding, self.candidate_embedding)



        loss_ans = self.bce(pred, ans_label)  # self.kge_loss(pred, ans_label)
        # calculate acc

        #r_score = torch.tensor(
            #kg.relation_matching(possible_relations, self.head_words, self.sigmoid(pred))).to(dev)
        sigmoid_pred = (self.sigmoid(pred))
        topk = 3
        local_count = 0
        acc = 0
        acc1 = 0
        values, indices = torch.topk(sigmoid_pred, topk)
        for item in zip(values, indices):
            for index, i in enumerate(item[1]):
                i = i.item()
                if ans_label[local_count][i] == 1:
                    acc1 += 1
                    if index==0:
                        acc += 1
                    break
        # for i in torch.argmax(sigmoid_pred, 1):
        #     if ans_label[local_count][i] == 1:
        #         acc += 1
            local_count += 1
        acc = acc / local_count
        acc1 = acc1 / local_count
        self.total_val_acc += acc
        self.total_val_acc1 += acc1


        # TODO reset

        self.head_coord_list = torch.tensor([[0, 0]]).to(dev)
        self.head_words = []
        self.head_embedding_list = torch.tensor([]).to(dev)

        self.total_val_loss += loss_ans.item()
        self.val_counter += 1
        return loss_ans  # {"loss": loss}

    def validation_epoch_end(self, validation_step_outputs):
        self.log("total_val_loss", self.total_val_loss / self.val_counter, prog_bar=True, logger=True)
        self.log("top1_acc", self.total_val_acc / self.val_counter, prog_bar=True, logger=True)
        self.log("cls_acc", self.cls_acc / self.val_counter, prog_bar=True, logger=True)
        print(self.total_val_loss / self.val_counter)
        print("top1_acc", self.total_val_acc / self.val_counter)
        print("top3_acc", self.total_val_acc1 / self.val_counter)
        print("cls_acc", self.cls_acc/ self.val_counter)
        self.total_val_loss = 0
        self.val_counter = 0
        self.total_val_acc = 0
        self.total_val_acc1 = 0
        self.cls_acc = 0

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
        optimizer1 = torch.optim.Adam(self.parameters(), lr=0.00002)
        optimizer2 = torch.optim.Adam(self.parameters(), lr=0.00002)

        return [optimizer1, optimizer2]# ,[scheduler1,scheduler2]


df = prepare_data()
print(df.shape)
val_df, train_df = train_test_split(df, test_size=0.9)#,shuffle=False)
print(train_df.shape, val_df.shape)
# BERT_MODEL_NAME = "bert-base-cased"
BERT_MODEL_NAME = "sentence-transformers/bert-base-nli-mean-tokens"
# tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(BERT_MODEL_NAME)

print(tokenizer)
"""load embeddings"""
from EnhancedKGQA_main.all_kg_fixed_3 import KG, Complex

kg = KG()
complex = Complex()
entity_embeddings, relation_embeddings, entity_dict, relation_dict = kg.load_kg_embeddings()

# TODO 先获取所有的candidtae embeddings 训练时候用， 一直保持不变
pretrained_embeddings, _ = kg.get_candidates_embeddings(
    k=0)  # when k=0 find all entities in KG, else find head k neighbours
candidate_embedding = nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=0).to('cuda')
print("entity_embeddings", entity_embeddings.shape)
# print("relation_embeddings", relation_embeddings.shape)

# relation_matrix, _ = kg.get_all_relations_embeddings()
relation_embedding = torch.FloatTensor(relation_embeddings).to('cuda')#.transpose(1,0)
print("relation_embedding", relation_embedding.shape)

N_EPOCHS = 300
BATCH_SIZE = 64
SMOOTH = 0
data_module = DataModule(entity_dict, train_df[:], val_df[:7680], tokenizer, BATCH_SIZE,64)
data_module.setup()

model = EntityPredictor(
    n_classes=2,
    relation_embedding=relation_embedding,
    candidate_embedding=candidate_embedding,
    entity_dict=entity_dict,
    steps_per_epoch=len(train_df) // BATCH_SIZE,
    n_epochs=N_EPOCHS

)
trainer = pl.Trainer(max_epochs=N_EPOCHS, gpus=1, progress_bar_refresh_rate=10, checkpoint_callback=False,check_val_every_n_epoch=1)
trainer.fit(model, data_module)


