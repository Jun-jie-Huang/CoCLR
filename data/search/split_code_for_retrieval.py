import json
import os
import re
import copy

def sep_docstring(snippet):
    # pos1 = list(re.finditer("\"\"\"", snippet))[0].regs[0][0]
    # pos2 = list(re.finditer("\"\"\"", snippet))[1].regs[0][1]
    # doc = snippet[pos1:pos2]
    # code = snippet[:pos1] + snippet[pos2:]
    flag = False
    try:
        pos1 = list(re.finditer("\"\"\"", snippet))[0].regs[0][0]
        pos2 = list(re.finditer("\"\"\"", snippet))[0].regs[0][1]
        pos3 = list(re.finditer("\"\"\"", snippet))[1].regs[0][0]
        pos4 = list(re.finditer("\"\"\"", snippet))[1].regs[0][1]
    except:
        print("code wrong! ")
        return "", "", True
    header = snippet[:pos1]
    doc = snippet[pos2:pos3]
    statement = snippet[pos4:]
    no_header = snippet[pos1:]
    no_documentation = snippet[:pos1] + snippet[pos4:]
    no_body = snippet[:pos4]
    return header, doc, statement, no_header, no_documentation, no_body
    # if len(doc) == 0:
    #     flag = True
    # if doc.isspace():
    #     flag = True
    # return doc, code, flag


test_file = "cosqa-retrieval-test-500.json"
with open(test_file, 'r', encoding='utf-8') as fp:
    test_data = json.load(fp)
code_base_file = "code_idx_map.txt"
with open(code_base_file, 'r') as f:
    codes = json.loads(f.read())
print(len(test_data))
print(len(codes))

# idx = 5
dict_map = {0: "header_only",
            1: "doc_only",
            2: "body_only",
            3: "no_header",
            4: "no_doc",
            5: "no_body"}
test_answers_idxs = set([inst['code'] for inst in test_data])
print(len(test_answers_idxs))  # 473


# seperated = sep_docstring(test_data[0]['code'])
# print("*************")
# print(seperated[0])
# print("*************")
# print(seperated[1])
# print("*************")
# print(seperated[2])
# print("*************")
# print(seperated[3])
# print("*************")
# print(seperated[4])
# print("*************")
# print(seperated[5])
# print("*************")
for idx in range(6):
    new_test_data = []
    for inst in test_data:
        seperated = sep_docstring(inst['code'])
        new_inst = copy.deepcopy(inst)
        new_inst['code'] = seperated[idx]
        new_test_data.append(new_inst)
    new_code_idx_map = {}
    for c, c_id in codes.items():
        seperated = list(sep_docstring(c))
        while seperated[idx] in new_code_idx_map:
            seperated[idx] = seperated[idx] + '\n'
        new_code_idx_map[seperated[idx]] = c_id
    print(len(new_code_idx_map))
    print(len({c_id:c for c, c_id in new_code_idx_map.items()}))

    # check idx match or not
    print("******")
    for inst in test_data:
        if inst['retrieval_idx'] != codes[inst['code']]:
            print(inst['retrieval_idx'])
            print(new_code_idx_map[inst['code']])
            print(inst['code'])
    #         exit()
    # for inst in new_test_data:
    #     if inst['retrieval_idx'] != new_code_idx_map[inst['code']]:
    #         print(inst['retrieval_idx'])
    #         print(new_code_idx_map[inst['code']])
    #         print(inst['code'])
    #         print()
    #         exit()

    print(len(new_code_idx_map))
    print(len(new_test_data))
    directory = "./ablation_test_code_component/"+dict_map[idx]
    if not os.path.exists(directory):
        os.makedirs(directory)
    with open(os.path.join(directory, 'code_idx_map.txt'), 'w', encoding='utf-8') as fp:
        fp.write(json.dumps(new_code_idx_map, indent=1))
    with open(os.path.join(directory, 'cosqa-retrieval-test-500.json'), 'w', encoding='utf-8') as fp:
        json.dump(new_test_data, fp, indent=1)

