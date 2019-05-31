from model.POS_CNN_CRF import utils

def get_state_list(train_file):
	sents_list, tags_list = utils.read_data(train_file)
	state_list = []
	for tags_tuple in tags_list:
		list1 = list(tags_tuple)
		state_list.extend(list1)
	state_list = list(set(state_list))
	# print ('HMM model state list: ',state_list)
	return state_list

if __name__ == '__main__':

	train_file = '../../data/trainset/train_pos_format'
	model_file = 'hmm_pos_model.pkl'
	test_file = '../../data/testset1/test_pos1_new.txt'
	test_result_file = '../../data/testset1/test_pos1_hmm.txt'
	ref_file = '../../data/testset1/test_pos1.txt'
	cad_file =  test_result_file

	state_list = get_state_list(train_file)
	print(state_list)
	print(len(state_list))


	id2tag = {0: 's', 1: 'b', 2: 'm', 3: 'e'}  # 标签（sbme）与id之间的映射
	tag2id = {j: i for i, j in id2tag.items()}

	id2tag = {}
	for i in range (0,len(state_list)):
		id2tag[i] = state_list[i]
	print(id2tag)
	print(type(tag2id))