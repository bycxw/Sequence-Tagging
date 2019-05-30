# coding: utf-8

#****************************************************************
#
# dataset.py
#
# POS的数据预处理，统一数据格式
#
#****************************************************************


import traceback

def format_pos_data(data_file, format_data_file):
	rf = open(data_file, 'r', encoding='utf-8')
	wf = open(format_data_file, 'w', encoding='utf-8')
	for line in rf:
		try:
			wf.write('\n')
			# 将原文本中的行与行，用'\n'来分隔
			line = line.strip()
			line_list = line.split()
			for unit in line_list:
				# print (unit)
				unit_list = unit.split('/')
				# print (unit_list)

				# 考虑到训练集中有噪声，有些分词后漏标了词性，这里只取有词性标注的，即分割后list长度是2的
				if len(unit_list)==2:
					word = unit_list[0]
					word_label = unit_list[1]
					wf.write(word + '\t' + word_label + '\n')

		except Exception as e:
			traceback.print_exc()
			break
	wf.close()
	rf.close()

if __name__ == '__main__':
	train_data_file = '../../data/trainset/train_pos.txt'
	train_format_data_file = '../../data/trainset/train_pos_format'
	format_pos_data(train_data_file, train_format_data_file)

	val_data_file = '../../data/devset/val_pos.txt'
	val_format_data_file = '../../data/devset/val_pos_format'
	format_pos_data(val_data_file, val_format_data_file)

	test_data_file = '../../data/testset1/test_pos1.txt'
	test_format_data_file = '../../data/testset1/test_pos1_format'
	format_pos_data(test_data_file, test_format_data_file)

	print ('POS数据格式预处理完成')




