import os

dataset = 'Movielens1M'

folder = 'datasets/%s/graphs' %dataset

entities = []

relations = []

with open('datasets/%s/KB2E/train.txt' %dataset, 'w') as write_kg:

	for file in os.listdir(folder):

		if 'edgelist' in file:

			prop_name = file.replace('.edgelist','')

			print(prop_name)

			with open('%s/%s' %(folder,file), 'r') as edgelist_read:

				for edge in edgelist_read:

					edge_split = edge.strip('\n').split(' ')

					left_edge = edge_split[0]

					right_edge = edge_split[1]

					write_kg.write('%s\t%s\t%s\n' %(left_edge, right_edge, prop_name))

					entities.append(left_edge)

					entities.append(right_edge)

					relations.append(prop_name)

# create index 

entities = list(set(entities))

with open('datasets/%s/KB2E/entity2id.txt' %dataset, 'w') as entity2id:

	for i, entity in enumerate(entities):

		entity2id.write('%s\t%d\n' %(entity, i))

relations = list(set(relations))

with open('datasets/%s/KB2E/relation2id.txt' %dataset, 'w') as relation2id:

	for i, relation in enumerate(relations):

		relation2id.write('%s\t%d\n' %(relation, i))

