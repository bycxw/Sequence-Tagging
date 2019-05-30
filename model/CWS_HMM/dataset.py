# coding: utf-8

#****************************************************************
#
# dataset.py
#
# CWS的数据预处理，统一数据格式
#
#****************************************************************


import traceback

def make_label(text):
	out_text = []
	if len(text) == 1:
		out_text = ['S']
	else:
		out_text += ['B'] + ['M'] * (len(text) - 2) + ['E']
	return out_text

def format_cws_data(data_file, format_data_file):
	rf = open(data_file, 'r', encoding='utf-8')
	wf = open(format_data_file, 'w', encoding='utf-8')
	for line in rf:
		try:
			wf.write('\n')
			# 将原文本中的行与行，用'\n'来分隔
			line = line.strip()
			line_list = line.split()
			for word in line_list:
				word_label = make_label(word)
				word = list(word)
				for i in range (0, len(word)):
					wf.write(word[i] + '\t' + word_label[i] + '\n')
		except Exception as e:
			traceback.print_exc()
			break
	wf.close()
	rf.close()


if __name__ == '__main__':
	train_data_file = '../../data/trainset/train_cws.txt'
	train_format_data_file = '../../data/trainset/train_cws_format'
	format_cws_data(train_data_file, train_format_data_file)

	val_data_file = '../../data/devset/val_cws.txt'
	val_format_data_file = '../../data/devset/val_cws_format'
	format_cws_data(val_data_file, val_format_data_file)

	test_data_file = '../../data/testset1/test_cws1.txt'
	test_format_data_file = '../../data/testset1/test_cws1_format'
	format_cws_data(test_data_file, test_format_data_file)

	print ('数据格式预处理完成')




