import sys

def format_data(data_file):
    pass

def ner_nest_layer(data_file):
    max_layer = 1
    cur_layer = 0
    left_cnt = 0
    right_cnt = 0
    with open(data_file, 'r', encoding='utf-8') as fr:
        for line in fr:
            try:
                line = list(line.strip())
                if cur_layer != 0:
                    print(line)
                    break
                for char in line:
                    if char == '[':
                        left_cnt += 1
                        cur_layer += 1
                        if cur_layer > max_layer:
                            max_layer = cur_layer
                            print(line)
                    elif char == ']':
                        cur_layer -= 1
                        right_cnt += 1
            except Exception as e:
                print(e.message)
                
    print(max_layer)
    print(cur_layer)
    print(left_cnt)
    print(right_cnt)

if __name__ == '__main__':
    data_file = sys.argv[1]
    ner_nest_layer(data_file)
                
