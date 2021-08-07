import json

with open("./qa/cosqa-20604.json", 'r') as fp:
    data = json.load(fp)

print("# pairs: {}".format(len(data)))
print("# label '1': {}".format(len([inst for inst in [inst['label'] for inst in data] if inst == 1])))
print("# label '0': {}".format(len([inst for inst in [inst['label'] for inst in data] if inst == 0])))
print("% label '1': {}".format(len([inst for inst in [inst['label'] for inst in data] if inst == 0]) / len(data)))

query = [inst['doc'] for inst in data]
code = [inst['code'] for inst in data]
docstring = [inst['docstring_tokens'] for inst in data]

print("# query: {}".format(len(set(query))))
print("avg. length: {}".format(sum([len(inst.split(' ')) for inst in query]) / len(query)))
print("# of tokens: {}".format(len(set([item for inst in query for item in inst.split(' ')]))))

print("# code: {}".format(len(set(code))))
print("avg. length(20604): {}".format(sum([len(inst.split(' ')) for inst in code]) / len(code)))
print("# of tokens(20604): {}".format(len(set([item for inst in code for item in inst.split(' ')]))))
code = list(set(code))
print("avg. length(6267) : {}".format(sum([len(inst.split(' ')) for inst in code]) / len(code)))
print("# of tokens(6267) : {}".format(len(set([item for inst in code for item in inst.split(' ')]))))


