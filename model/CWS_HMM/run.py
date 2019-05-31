# coding: utf-8


from model.CWS_HMM import utils
from model.CWS_HMM import hmm_model
from model.CWS_HMM import evaluate


def get_state_list(train_file):
	sents_list, tags_list = utils.read_data(train_file)
	state_list = []
	for tags_tuple in tags_list:
		list1 = list(tags_tuple)
		state_list.extend(list1)
	state_list = list(set(state_list))
	print ('HMM model state list: ',state_list)
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

	data_seg = []
	for line in data:
		str = ' '.join(hmm.cut(line))
		data_seg.append(str)

	result_file = test_result_file
	f2 = open(result_file, 'w')
	f2.writelines(data_seg)
	f2.close()

	print ('model test done, results save to ',test_result_file)

def model_evaluate(ref_file,cad_file):
	precision, recall, f1 = evaluate.evaluate(ref_file,cad_file)
	print('model evaluation done.')
	return precision, recall, f1

if __name__ == '__main__':

	train_file = '../../data/trainset/train_cws_format'
	model_file = 'hmm_model.pkl'
	test_file = '../../data/testset1/test_cws1_new.txt'
	test_result_file = '../../data/testset1/test_cws1_hmm.txt'
	ref_file = '../../data/testset1/test_cws1.txt'
	cad_file =  test_result_file

	# state_list = get_state_list(train_file)
	state_list = ['B','M','E','S']
	hmm_model_run(train_file,model_file,state_list)
	hmm_model_test(test_file,test_result_file,state_list)
	precision, recall, f1 = model_evaluate(ref_file, test_result_file)
	print ('precision:',precision)
	print ('recall:',recall)
	print ('f1 score:',f1)