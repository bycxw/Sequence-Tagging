
'''
提供的测试文件test_cws1.txt是已经分好词带空格的
所以需要去除空格，构建尚未分词的测试语料
'''

input_file = '../../data/testset1/test_cws1.txt'
f = open(input_file,"r")   #设置文件对象
raw_data = f.readlines()  #直接将文件中按行读到list里，效果与方法2一样
print (raw_data[0])
print (len(raw_data))
f.close()



new_data = []
for i in raw_data:
	i = i.replace(' ','')
	new_data.append(i)


output_file = '../../data/testset1/test_cws1_new.txt'
f2 = open (output_file,"w")
f2.writelines(new_data)
f2.close()

print (new_data[0])
print (len(new_data))
