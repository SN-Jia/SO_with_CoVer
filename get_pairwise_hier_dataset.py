import sys, random, itertools
from collections import defaultdict

dataset = sys.argv[1]

data_type = ['train', 'val', 'test']

for item in data_type:
    pn_data = defaultdict(list)
    total_id = 0

    with open(f'../data/so/{dataset}/{item}.lower') as f:
        for id, line in enumerate(f):
            doc = line.strip().split(' @eos@ ')
            if len(doc) < 4:
                continue
            gold_order = [i for i in range(len(doc))]

            gold_order = gold_order[:-2]

            sent_num = len(doc)
            permu_cnt = 1 # Permutation can be done multiple times

            all_pairs = []
            for i in range(permu_cnt):
                pn_data[total_id].append(doc)

                # First permutation
                current_pair = random.sample(gold_order, 2)
                while set(current_pair) in all_pairs:
                    current_pair = random.sample(gold_order, 2)
                all_pairs.append(set(current_pair))

                new_doc = doc.copy()
                new_doc[current_pair[0]], new_doc[current_pair[1]] = new_doc[current_pair[1]], new_doc[current_pair[0]]

                pn_data[total_id].append(new_doc.copy())

                new_order = gold_order.copy()
                new_order.remove(current_pair[0])
                new_order.remove(current_pair[1])

                # Next permutation
                while len(new_order) > 1:
                    current_pair = random.sample(new_order, 2)
                    while set(current_pair) in all_pairs:
                        current_pair = random.sample(gold_order, 2)
                    all_pairs.append(set(current_pair))

                    new_order.remove(current_pair[0])
                    new_order.remove(current_pair[1])

                    new_doc[current_pair[0]], new_doc[current_pair[1]] = new_doc[current_pair[1]], new_doc[current_pair[0]]

                    pn_data[total_id].append(new_doc.copy())

                total_id += 1

            
    

    with open(f'../data/hier/{dataset}/{item}.lower', 'w') as fw:
        for k, v in pn_data.items():
            # Two negative samples for one document
            choice_ids = random.sample([i for i in range(1, len(v))], 2)

            fw.write(f'{k}\t{v[0]}') # positive sample
            for c in choice_ids:
                fw.write(f'{k}\t{v[c]}\n')