filename = './worker_long.log'
fobj = open(filename, 'r')
import re
trail_random_success = 0
trail_random_fail = 0
trail_heredity_layer = 0
trail_mutation_edge = 0
trail_mutation_layer = 0
trail_mutation_type = 0
early_stop = 0

for fileLine in fobj:
    if "random op" in fileLine.lower() and "Success" in fileLine:
        a = re.findall(r"\d*", fileLine.lower())
        for i in reversed(a):
            if i.isdigit():
                trail_random_success = trail_random_success + int(i)
                break
    if "random op" in fileLine.lower() and "failed" in fileLine.lower():
        a = re.findall(r"\d*", fileLine.lower())
        for i in reversed(a):
            if i.isdigit():
                trail_random_fail = trail_random_fail + int(i)
                break
    if "trail" in fileLine.lower():
        if "mutation" in fileLine.lower() and "edge" in fileLine.lower():
            a = re.findall(r"\d*", fileLine.lower())
            for i in reversed(a):
                if i.isdigit():
                    trail_mutation_edge = trail_mutation_edge+int(i)
                    break
        elif "mutation" in fileLine.lower() and "layer" in fileLine.lower():
            a = re.findall(r"\d*", fileLine.lower())
            for i in reversed(a):
                if i.isdigit():
                    trail_mutation_layer = trail_mutation_layer+int(i)
                    break
        elif "mutation" in fileLine.lower() and "type" in fileLine.lower():
            a = re.findall(r"\d*", fileLine.lower())
            for i in reversed(a):
                if i.isdigit():
                    trail_mutation_type = trail_mutation_type+int(i)
                    break
        elif "heredity" in fileLine.lower() and "layer" in fileLine.lower():
            a = re.findall(r"\d*", fileLine.lower())
            for i in reversed(a):
                if i.isdigit():
                    trail_heredity_layer = trail_heredity_layer+int(i)
                    break
    elif "early stop" in fileLine.lower():
        early_stop = early_stop+1

print("trail_heredity_layer: ",trail_heredity_layer)
print("trail_mutation_type: ",trail_mutation_type)
print("trail_mutation_edge: ",trail_mutation_edge)
print("trail_mutation_layer: ",trail_mutation_layer)
print("trail_random_success: ",trail_random_success)
print("trail_random_failed: ",trail_random_fail)
print("early_stop: ",early_stop)