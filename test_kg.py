from all_kg import KG

kg = KG()

question = []
path = './data/QA_data/MetaQA/qa_train_2hop.txt'
with open(path, 'r', encoding='utf-8') as rf:
    data = rf.readlines()
    for r in data:
        line = r.split('\t')
        q = line[0].strip()
        a = line[1].strip().split('|')
        start = q.index('[')
        end = q.index(']')      
        head = q[start+1:end]
        # q_clean = q[:start] + q[start+1:end] + q[end+1:]
        # print(q_clean)
        # print('span:', kg.find_span_question_exhaust(q_clean))
        # neighbors = kg.dfs_neighbours(head, 2)
        # if len(list(set(a) & set(neighbors))) == 0:
        #     print(q)
        

        


