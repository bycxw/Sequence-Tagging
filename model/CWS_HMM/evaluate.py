def data_compare(gold_data, cad_data):
	gold_count = 0
	cad_count = 0
	positive_count =0
	for g_line,c_line in zip(gold_data,cad_data):
		g_line_seg = g_line.split(' ')
		c_line_seg = c_line.split(' ')
		gold_count += len(g_line_seg)
		cad_count += len(c_line_seg)
		correct_seg =  [i for i in g_line_seg if i in c_line_seg]
		positive_count += len(correct_seg)
		# print (len(g_line_seg), len(c_line_seg), len(correct_seg))

	recall = positive_count/gold_count
	precision = positive_count/cad_count
	if recall + precision == 0:
		f1 = 0
	else:
		f1 = 2 * recall * precision / (recall + precision)
	return precision, recall, f1

def evaluate(ref_file,cad_file):
	refs = open(ref_file,encoding='utf-8').readlines()
	cads = open(cad_file, encoding='utf-8').readlines()
	precision, recall, f1 = data_compare(refs, cads)
	return precision, recall, f1


if __name__ == '__main__':
	refs = open('../../data/testset1/test_cws1.txt', encoding='utf-8').readlines()
	cads = open('../../data/testset1/test_cws1_hmm.txt', encoding='utf-8').readlines()
	precision, recall, f1 = data_compare(refs, cads)
	print({'precision': precision, 'recall': recall, 'f1': f1})



