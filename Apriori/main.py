from efficient_apriori import apriori
import os
import psutil
import time
import itertools


def get_memory():
    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    memory = info.uss / 1024
    return memory


def transactions_from_file(filename):
    with open(filename) as file:
        for line in file:
            yield tuple(k.strip() for k in line.split(","))
try:
    base, _ = os.path.split(__file__)
    filename = os.path.join(base, "efficient_apriori/tests/adult_data_cleaned.txt")
except NameError:
    filename = "efficient_apriori/tests/adult_data_cleaned.txt"
transactions = transactions_from_file(filename)

MIN_SUPP = [0.05, 0.1, 0.15]
MIN_CONF = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]

min_supp = MIN_SUPP[2]
min_conf = MIN_CONF[5]
total_time, total_memory = 0.0, 0.0

start_time = time.perf_counter()
start_memory = get_memory()
itemsets, rules = apriori(transactions, min_support=min_supp, min_confidence=min_conf)
end_memory = get_memory()
end_time = time.perf_counter()

total_memory += (end_memory - start_memory)
total_time += (end_time - start_time) * 1000
print(round(total_memory, 4), 'KB')
print(round(total_time, 4), 'ms')

item_count = 0
for i, itemset in itemsets.items():
    item_count += len(itemset)

rule_count = len(rules)

print('item number:', item_count)
print('rule number:', rule_count)

for rule in rules[400:]:
    print(rule)