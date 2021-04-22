import numpy as np
import random
import torch
import networkx as nx

class KG():
    def __init__(self, path='./EnhancedKGQA_main/ComplEx_MetaQA_full/'):
        self.r_em_path = path + 'R.npy'
        self.e_em_path = path + 'E.npy'
        self.r_dict_path = path + 'relations.dict'
        self.e_dict_path = path + 'entities.dict'

        self.entity_embeddings, self.relation_embeddings, self.entity_dict, self.relation_dict = self.load_kg_embeddings()

        self.graph1 = nx.MultiDiGraph()
        self.build_graph('./EnhancedKGQA_main/data/MetaQA/kb.txt')

        self.num_entity = len(self.entity_dict)
        self.num_relation = len(self.relation_dict)
        self.id2e = list(self.entity_dict.keys())
        self.id2r = list(self.relation_dict.keys())
        self.neighbour_cache = {}  # {'head': {k: [0,0,0....]}}
        self.path_relation_cache = {}
        self.path_relation_all_cache = {}

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

    def build_graph(self, path):
        with open(path, 'r', encoding='utf-8') as rf:
            lines = rf.readlines()
            for l in lines:
                triple = l.strip().split('|')
                head = triple[0]
                relation = self.relation_dict[triple[1]]
                tail = triple[2]

                # Define: reverse relation = relation + 1
                self.graph1.add_edge(head, tail, weight=relation)
                self.graph1.add_edge(tail, head, weight=relation + 1)

    def dfs_neighbours(self, node, depth):
        return nx.single_source_shortest_path_length(self.graph1, node, cutoff=depth).keys()

    def get_path(self, start, goal):
        # get a single short path -> list
        return nx.shortest_path(self.graph1, start, goal)

    def get_path_fixed_length(self, start, goal, k):
        try:
            paths = list(nx.all_simple_paths(self.graph1, start, goal, cutoff=k))
            paths = [p for p in paths if len(p) == k+1]
        except:
            print('No path between %s and %s by length %d.' % (start, goal, k))
            paths = list()
        return paths

    def get_all_path(self, start, goal):
        # get list of short path -> list[list]
        return nx.all_shortest_paths(self.graph1, start, goal)

    def get_all_relations_embeddings(self):
        relation_matrix = []
        relations = self.relation_dict.keys()
        for re in relations:
            r = self.relation_embeddings[self.relation_dict[re]]
            relation_matrix.append(r)
        return relation_matrix, list(relations)

    def get_candidates_embeddings(self, k, head=None):
        candidate_matrix = []
        if k == 0:
            neighbors = self.entity_dict.keys()
        elif k > 0 and head != None:
            neighbors = self.dfs_neighbours(head, k)
        else:
            raise Exception('Head is not specified.')
        for n in neighbors:
            e = self.entity_embeddings[self.entity_dict[n]]
            candidate_matrix.append(e)
        return candidate_matrix, list(neighbors)

    def get_neighbour_onehot(self, heads, ks, smoothing=0):
        # return a batch of heads and ks neighbour matrix
        neighbour_matrix = []
        for i, head in enumerate(heads):
            this_k = ks[i]
            key = '->'.join([head, str(this_k)])
            try:
                this_one_hot = self.neighbour_cache[key]
            except:
                neighbour = self.dfs_neighbours(head, this_k)
                indices = []
                for n in neighbour:
                    indices.append(self.entity_dict[n])
                this_one_hot = np.zeros(self.num_entity)
                this_one_hot[indices] = 1
                # caching
                self.neighbour_cache[key] = this_one_hot
            # smoothing
            if smoothing:
                this_one_hot = ((1.0 - smoothing) * this_one_hot) + (smoothing / this_one_hot.shape[0])
            neighbour_matrix.append(this_one_hot)

        return neighbour_matrix

    def find_span_question_exhaust(self, q):
        spans = []
        q_split = q.split()
        l = len(q_split)
        for s in range(l):
            for e in range(s + 1, l + 1):
                span = ' '.join(q_split[s:e])
                if span in self.entity_dict.keys():
                    spans.append(span)
        return spans

    def path_relations(self, start, goal, k):
        # return relations between head and tail
        # all possibility, target sampled from those at each epoch
        try:
            sps = self.path_relation_cache['->'.join([start, goal, str(k)])]
        except:
            if k != 0:
                sps = self.get_path_fixed_length(start, goal, k)
            else:
                sps = self.get_path(start, goal)
            self.path_relation_cache['->'.join([start, goal, str(k)])] = sps

        relation_onehot = np.zeros(self.num_relation)
        try:
            if k !=0:
                sp = random.choice(sps)
                pathGraph = nx.path_graph(sp)
            else:
                pathGraph = nx.path_graph(sps)
            for ea in pathGraph.edges():
                r = self.graph1.edges[ea[0], ea[1], 0]['weight']
                # index = self.relation_dict[r]
                index = r
                relation_onehot[index] = 1
        except:
            pass
        return relation_onehot

    def path_relations_batch(self, starts, goals_onehot, k):
        # target for learning cls token
        paths = []
        for i, start in enumerate(starts):
            index = torch.argmax(goals_onehot[i]).item()  # find one path for now
            paths.append(self.path_relations(start, self.id2e[index], k))
        return np.stack(paths)

    def find_tails_from_head(self, current, paths, result, depth=0):
        # beam with k = current relation
        if depth == len(paths):
            result.append(current)
            return
        neighbors_dict = self.graph1[current]  # find neighbours -> dict
        refined_neighbors = []
        for k, v in neighbors_dict.items():
            for _, v1 in v.items():
                if v1['weight'] == paths[depth]:
                    refined_neighbors.append(k)
                    continue
        for c in refined_neighbors:
            self.find_tails_from_head(c, paths, result, depth + 1)
        return

    def candidates2onehot(self, candidates):
        indices = []
        for n in candidates:
            indices.append(self.entity_dict[n])
        this_one_hot = np.zeros(self.num_entity)
        this_one_hot[indices] = 1
        return this_one_hot

    # relation matrching
    def rm_score(self, start, goal, relation, k=3):
        try:
            sps = self.path_relation_cache['->'.join([start, goal,str(k)])]
        except:
            try:
                sps = self.get_path(start, goal)
                self.path_relation_cache['->'.join([start, goal,str(k)])] = sps
            except:
                return 0
        
        score = 0

        try:
            relation_list = []
            for sp in sps:
                pathGraph = nx.path_graph(sp)
                for ea in pathGraph.edges():
                    r = self.graph1.edges[ea[0], ea[1], 0]['weight']
                    relation_list.append(r)
            score = len(list(set(relation_list) & set(relation)))
        except:
            pass
        return score

    # relation matrching
    def relation_matching(self, relations, starts, candidates):
        # relations: [[index, index, ...], [index, index, ...]]
        rel_scores = []
        #print("##################")
        for i, start in enumerate(starts): # for each batch
            candidate = candidates[i]
            relation = relations[i]
            relation = torch.nonzero(relation).squeeze(1).tolist()
            #print(relation)
            _, indices = torch.topk(candidate, 10) # only consider top10

            rel_score = np.zeros(self.num_entity)
            for index in indices:
                score = self.rm_score(start, self.id2e[index.item()], relation)
                rel_score[index.item()] = score   

            rel_scores.append(rel_score)

        return np.stack(rel_scores)

    
class Complex():
    def __init__(self, reldrop=0.0, entdrop=0.0, scoredrop=0.0, batch_norm=False, device='cuda'):
        self.bn_list = []
        for i in range(3):
            bn = np.load('./EnhancedKGQA_main/ComplEx_MetaQA_full/bn' + str(i) + '.npy', allow_pickle=True)
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

        self.rel_dropout = torch.nn.Dropout(reldrop)
        self.ent_dropout = torch.nn.Dropout(entdrop)
        self.score_dropout = torch.nn.Dropout(scoredrop)
        self.batch_norm = batch_norm

    def score(self, head, relation, candidate):
        head = torch.stack(list(torch.chunk(head, 2, dim=1)), dim=1)
        if self.batch_norm:
            head = self.bn0(head)

        head = self.ent_dropout(head)
        relation = self.rel_dropout(relation)

        head = head.permute(1, 0, 2)
        re_head = head[0]
        im_head = head[1]

        re_relation, im_relation = torch.chunk(relation, 2, dim=1)
        re_tail, im_tail = torch.chunk(candidate.weight, 2, dim =1)

        re_score = re_head * re_relation - im_head * im_relation
        im_score = re_head * im_relation + im_head * re_relation

        score = torch.stack([re_score, im_score], dim=1)
        if self.batch_norm:
            score = self.bn2(score)

        score = self.score_dropout(score)
        score = score.permute(1, 0, 2)

        re_score = score[0]
        im_score = score[1]
        score = torch.mm(re_score, re_tail.transpose(1,0)) + torch.mm(im_score, im_tail.transpose(1,0))
        # pred = torch.sigmoid(score)
        return score

    def get_top_k(self, scores, k):
        return torch.topk(scores, k, largest=True, sorted=True)
        

# test
# kg = KG()

# sps = kg.get_path_fixed_length('John A. Davis','Nicolas Cage', 2)
# for sp in sps:
#     pathGraph = nx.path_graph(sp)
#     for ea in pathGraph.edges():
#         r = kg.graph1.edges[ea[0], ea[1], 0]['weight']
#         print(r,end=' ')
#     print()

# head = 'Send Me No Flowers'
# tail = '1983'
# k = 1
# path = kg.get_path(head, tail)
# print('Hop: [%s] -> [%s] = %d' % (head, tail, len(path)-1))
# print(path)
# neighbors = kg.dfs_neighbours(head, k)
# print(len(neighbors))
# # print('Heighbours from [%s] with k = %d: %s' % (head, k, neighbors.__str__()))
# if tail in neighbors:
#     print('YES')
# else:
#     print('NO')

# entity_embeddings, relation_embeddings, entity_dict, relation_dict = kg.load_kg_embeddings()

# head = 'Lloyd\'s of London'
# relation = 'starred_actors'

# this_r = torch.FloatTensor(relation_embeddings[relation_dict[relation]]).to('cuda')
# this_e = torch.FloatTensor(entity_embeddings[entity_dict[head]]).to('cuda')
# print(this_r.shape)
# pretrained_embeddings, candidate_dict = kg.get_candidates_embeddings(k=1, head=head) #when k=0 find all entities in KG, else find head k neighbours
# candidate_dict.remove(head)
# print(candidate_dict)
# import torch.nn as nn
# candidate_embedding= nn.Embedding.from_pretrained(torch.FloatTensor(pretrained_embeddings), freeze=True).to('cuda')

# complex = Complex('cuda')
# score = complex.score(this_e.unsqueeze(0), this_r.unsqueeze(0), candidate_embedding)
# print(score.shape)
# top_results = complex.get_top_k(score, 1)
# for _, idx in zip(top_results[0], top_results[1]):
#     print(candidate_dict[idx.item()])

# a = torch.tensor([[1,2,3],[2,4,3]])
# b = torch.tensor([[1,0,1],[0,1,1]])
# print(torch.mul(b,a))