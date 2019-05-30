# coding: utf-8


from model.POS_HMM import utils
from model.POS_HMM import hmm_model
from model.POS_HMM import pos_evaluate


def get_state_list(train_file):
	sents_list, tags_list = utils.read_data(train_file)
	state_list = []
	for tags_tuple in tags_list:
		list1 = list(tags_tuple)
		state_list.extend(list1)
	state_list = list(set(state_list))
	# print ('HMM model state list: ',state_list)
	return state_list
def hmm_model_run (train_file,model_file,state_list):
	sents_list, tags_list = utils.read_data(train_file)
	hmm = hmm_model.HMM(state_list=state_list)
	hmm.train(sents_list, tags_list, model_file)

def hmm_model_test (test_file,test_result_file,state_list):
	hmm = hmm_model.HMM(state_list=state_list)
	hmm.load_model(model_file)

	f = open(test_file, 'r')
	data = f.readlines()
	f.close()

	data_pos_predict = []
	for line in data:
		line = line.rstrip('\n')
		word_list = line.split(' ')
		# print(word_list)
		tag_list = hmm.pos_predict(word_list)
		unit_list = []
		for i in range (0,len(word_list)):
			unit = word_list[i] + '/' + tag_list[i]
			# print(unit)
			unit_list.append(unit)

		# print(unit_list)
		sentense = ' '.join(unit_list)
		sentense = sentense + '\n'
		# print(sentense)
		data_pos_predict.append(sentense)

	result_file = test_result_file
	f2 = open(result_file, 'w')
	f2.writelines(data_pos_predict)
	f2.close()
	print ('model test done, results save to ',test_result_file)

	# text = '大多数纵向研究着重于患儿的拒绝上学症状，年幼儿童早期发病的预后较好，一般能较早回到学校；而青少年儿童在发病时伴有其他症状如学习困难，则预后相对比年幼儿童差一些。作为一个群体来说，儿童患分离性焦虑症是成人期焦虑症的一个风险较大的高危因素。'
	# text = ['我','爱','你们','的','书包','，','你们','真','好']
	# print(' '.join(hmm.pos_predict(text)))



# def model_evaluate(ref_file,cad_file):
# 	precision, recall, f1 = evaluate.evaluate(ref_file,cad_file)
# 	print('model evaluation done.')
# 	return precision, recall, f1

if __name__ == '__main__':

	train_file = '../../data/trainset/train_pos_format'
	model_file = 'hmm_pos_model.pkl'
	test_file = '../../data/testset1/test_pos1_new.txt'
	test_result_file = '../../data/testset1/test_pos1_hmm.txt'
	ref_file = '../../data/testset1/test_pos1.txt'
	cad_file =  test_result_file

	state_list = get_state_list(train_file)
	print ('start to train ....')
	hmm_model_run(train_file,model_file,state_list)
	print('start to test ....')
	hmm_model_test(test_file,test_result_file,state_list)
	print('start to evaluate ....')
	refs = open(ref_file, encoding='utf-8').readlines()
	output = []
	cads = open(test_result_file, encoding='utf-8').readlines()
	word_precision, word_recall, word_fmeasure, tag_precision, tag_recall, tag_fmeasure, word_tnr, tag_tnr = pos_evaluate.score(refs,
																												   cads,
																												   verbose=True)
	print("Tag precision:", tag_precision)
	print("Tag recall:", tag_recall)
	print("Tag F-measure:", tag_fmeasure)