# coding: utf-8

import sys
import traceback
import re

pattern_alpha = re.compile(r'[a-z]')

def format_ner_data(data_file, format_data_file):
    '''
    IOB format
    '''

    rf = open(data_file, 'r', encoding='utf-8')
    wf = open(format_data_file, 'w', encoding='utf-8')
    for line in rf:
        try:
            wf.write('\n')
            line = list(line.strip())
            if line == "":
                continue
            ne_layer = 0
            ne_cache = [[] for i in range(4)]
            i = 0
            tag = 'X'
            left_bracket = 0
            while i < len(line):
                if line[i] == ' ':
                    i += 1
                    continue
                if line[i] == '[':
                    left_bracket += 1
                    ne_layer += 1
                    i += 1
                elif line[i] == ']':
                    end = 1
                    while i+end < len(line) and re.match(pattern_alpha, line[i+end]):
                        end += 1
                    tag = ''.join(line[i+1: i+end]).upper()
                    i += end
                    for j in range(len(ne_cache[ne_layer])):
                        if ne_cache[ne_layer][j] in ['B-', 'I-']:
                            ne_cache[ne_layer][j] += tag
                    ne_layer -= 1
                    left_bracket = 0
                else:
                    ne_cache[0].append(line[i])
                    for j in range(1, ne_layer+1-left_bracket):
                        ne_cache[j].append('I-')
                    for j in range(ne_layer+1-left_bracket, ne_layer+1):
                        ne_cache[j].append('B-')
                    for j in range(ne_layer+1, 4):
                        ne_cache[j].append('O')
                    i += 1
                    left_bracket = 0
            for i in range(len(ne_cache[0])):
                for j in range(1):
                    wf.write(ne_cache[j][i] + '\t')
                wf.write(ne_cache[3][i] + '\n')
        except Exception as e:
            traceback.print_exc()
            break
    wf.close()
    rf.close()



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

def test_raw_data(data_file, tags):
    with open(data_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                line = line.strip()
                if not line:
                    continue
                for i in range(len(line)):
                    if line[i] == ']':
                        end = 1
                        while i+end < len(line) and re.match(pattern_alpha, line[i+end]):
                            end += 1
                        tag = ''.join(line[i+1: i+end])
                        if tag not in tags:
                            print(line)
                            print(tag)
                            return
            except:
                print(line)
            


if __name__ == '__main__':
    data_file = sys.argv[1]
    format_data_file = sys.argv[2]
    # ner_nest_layer(data_file)
    format_ner_data(data_file, format_data_file)

    # tags = set(['tes', 'tre', 'dis', 'sym', 'bod', 'nt', 'nr'])
    # test_raw_data(data_file, tags)
