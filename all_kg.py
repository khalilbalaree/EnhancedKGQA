import numpy as np
import torch
import torch.nn as nn

class KG():
    def __init__(self, path = './ComplEx_MetaQA_full/'):
        self.r_em_path = path + 'R.npy'
        self.e_em_path = path + 'E.npy'
        self.r_dict_path = path + 'relations.dict'
        self.e_dict_path = path + 'entities.dict'

        self.graph = dict()
        self.build_graph('./data/MetaQA/train.txt')

        self.entity_embeddings, self.relation_embeddings, self.entity_dict, self.relation_dict = self.load_kg_embeddings()
    
    def load_kg_embeddings(self):
        relation_embeddings = np.load(self.r_em_path)
        entity_embeddings = np.load(self.e_em_path)
        
        relation_dict = {}
        f = open(self.r_dict_path, 'r', encoding='utf-8')
        for line in f:
            line = line.strip().split('\t')
            r_id = int(line[0])
            r_name = line[1]
            relation_dict[r_name] = r_id
        f.close()

        entity_dict = {}
        f = open(self.e_dict_path, 'r', encoding='utf-8')
        for line in f:
            line = line.strip().split('\t')
            ent_id = int(line[0])
            ent_name = line[1]
            entity_dict[ent_name] = ent_id
        f.close()

        return entity_embeddings, relation_embeddings, entity_dict, relation_dict
    
    def add_node(self, parent, child):
        try:
            temp = self.graph[parent]
            temp.add(child)
            self.graph[parent] = temp
        except:
            temp = set()
            temp.add(child)
            self.graph[parent] = temp 
    
    def build_graph(self, path):
        # relations = set()
        with open(path, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
            for l in lines:
                triple = l.strip().split('\t')
                head = triple[0]
                # relation = triple[1]
                tail = triple[2]
                # relations.add(relation)
                self.add_node(head, tail)
                self.add_node(tail, head)
    
    def dfs_neighbours(self, node, depth, current_depth=0, visited=set()):
        if current_depth > depth:
            return
        # if node not in visited:
        visited.add(node)
        for n in self.graph[node]:
            self.dfs_neighbours(n, depth, current_depth+1, visited)
        return visited

    def bfs_path(self, start, goal):
        explored = [] 
        queue = [[start]] 
        if start == goal: 
            return []
        while queue: 
            path = queue.pop(0) 
            node = path[-1] 
            if node not in explored: 
                neighbours = self.graph[node]
                for neighbour in neighbours: 
                    new_path = list(path) 
                    new_path.append(neighbour) 
                    queue.append(new_path)     
                    if neighbour == goal: 
                        return new_path
                explored.append(node) 
        return False
    
    def get_candidates_embeddings(self, head, k):
        candidate_matrix = []
        neighbors = self.dfs_neighbours(head, k)
        for n in neighbors:
            e = entity_embeddings[entity_dict[n]]
            candidate_matrix.append(e)
        return candidate_matrix, list(neighbors)


class Complex():
    def __init__(self, device='cpu'):
        self.bn_list = []
        for i in range(3):
            bn = np.load('./ComplEx_MetaQA_full/bn' + str(i) + '.npy', allow_pickle=True) 
            self.bn_list.append(bn.item())
        self.bn0 = torch.nn.BatchNorm1d(2)
        self.bn2 = torch.nn.BatchNorm1d(2)

        for i in range(3):
            for key, value in self.bn_list[i].items():
                self.bn_list[i][key] = torch.Tensor(value).to(device)

        self.bn0.weight.data = self.bn_list[0]['weight']
        self.bn0.bias.data = self.bn_list[0]['bias']
        self.bn0.running_mean.data = self.bn_list[0]['running_mean']
        self.bn0.running_var.data = self.bn_list[0]['running_var']

        self.bn2.weight.data = self.bn_list[2]['weight']
        self.bn2.bias.data = self.bn_list[2]['bias']
        self.bn2.running_mean.data = self.bn_list[2]['running_mean']
        self.bn2.running_var.data = self.bn_list[2]['running_var']

    def score(self, head, relation, candidate):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        head = self.bn0(head)
        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(candidate.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        score = self.bn2(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        pred = torch.sigmoid(score)
        return pred

    def get_top_k(self,scores, k):
        return torch.topk(scores, k, largest=True, sorted=True)
        

# test
kg = KG()
# head = 'Spencer\'s Mountain'
# tail = '1947'
# k = 3
# path = kg.bfs_path(head, tail)
# neighbors = kg.dfs_neighbours(head, k)
# print('Hop: [%s] -> [%s] = %d' % (head, tail, len(path)-1))
# neighbors.remove(head)
# print(len(neighbors))
# # print('Heighbours from [%s] with k = %d: %s' % (head, k, neighbors.__str__()))
# if tail in neighbors:
#     print('YES')
# else:
#     print('NO')

entity_embeddings, relation_embeddings, entity_dict, relation_dict = kg.load_kg_embeddings()

head = 'Lloyd\'s of London'
relation = 'starred_actors'
this_r = torch.FloatTensor(relation_embeddings[relation_dict[relation]])
this_e = torch.FloatTensor(entity_embeddings[entity_dict[head]])
pretrained_embeddings, candidate_dict = kg.get_candidates_embeddings(head, k=2)
candidate_embedding= nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=True)

complex = Complex()
score = complex.score(this_e.unsqueeze(0), this_r.unsqueeze(0), candidate_embedding)
top_results = complex.get_top_k(score, 1)
for _, idx in zip(top_results[0], top_results[1]):
    print(candidate_dict[idx.item()])