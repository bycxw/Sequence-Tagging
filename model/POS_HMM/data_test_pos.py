
'''
提供的测试文件test_pos1.txt是已经标好词性的
所以需要把词性去掉，构建尚未标注的测试语料
'''

input_file = '../../data/testset1/test_pos1.txt'
f = open(input_file,"r")   #设置文件对象
raw_data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
print (raw_data[3])
print (len(raw_data))
f.close()



new_data = []
for sentense in raw_data:
	unit_list = sentense.split(' ')
	word_list = []
	for unit in unit_list:
		unit_l = unit.split('/')
		word_list.append(unit_l[0])
	new_sentense = ' '.join(word_list)
	new_sentense = new_sentense + '\n'
	new_data.append(new_sentense)


output_file = '../../data/testset1/test_pos1_new.txt'
f2 = open (output_file,"w")
f2.writelines(new_data)
f2.close()

print (new_data[3])
print (len(new_data))
