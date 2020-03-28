from itertools import permutations

s = 100
file_path= "data/browsing.txt"
f = open(file_path, "r")

C1 = {}
for line in f:
  items = line.strip().split(" ")
  for item in items:
    if item not in C1: C1[item] = 1
    else: C1[item] += 1

L1 = {}
for item in C1:
  if C1[item] >= s: L1[item] = C1[item]

f.seek(0)

C2 = {}
for line in f:
  items = line.strip().split(" ")
  for i in range(0, len(items) - 1):
    for j in range(i + 1, len(items)):
      if items[i] in L1 and items[j] in L1:
        if items[i] < items[j]: key = (items[i], items[j])
        else: key = (items[j], items[i])

        if key not in C2: C2[key] = 1
        else: C2[key] += 1

L2 = {}
for item in C2:
  if C2[item] >= s: L2[item] = C2[item]

L2_confidence = []
for k, v in L2.items():
  L2_confidence.append((k[0], k[1], v / L1[k[0]]))
  L2_confidence.append((k[1], k[0], v / L1[k[1]]))

L2_confidence = sorted(L2_confidence, key = lambda x: -x[2])

print("----------(d)----------")
for i in range(5):
  print("{} => {}, confidence is {}".format(L2_confidence[i][0], L2_confidence[i][1], L2_confidence[i][2]))

f.seek(0)
C3 = {}
for line in f:
  items = line.strip().split(" ")
  for i in range(0, len(items) - 2):
    for j in range(i + 1, len(items) - 1):
      for k in range(j + 1, len(items)):
        perm = permutations([i, j, k])
        for p in list(perm):
          if items[p[0]] < items[p[1]] < items[p[2]]:
            pair1 = (items[p[0]], items[p[1]])
            pair2 = (items[p[0]], items[p[2]])
            pair3 = (items[p[1]], items[p[2]])
            if pair1 in L2 and pair2 in L2 and pair3 in L2:
              key = (items[p[0]], items[p[1]], items[p[2]])
              if key in C3: C3[key] += 1
              else: C3[key] = 1

L3 = {}
for key in C3:
  if C3[key] >= s:
    L3[key] = C3[key]

L3_confidence = []
for k, v in L3.items():
  L3_confidence.append(((k[0], k[1]), k[2], v / L2[(k[0], k[1])]))
  L3_confidence.append(((k[0], k[2]), k[1], v / L2[(k[0], k[2])]))
  L3_confidence.append(((k[1], k[2]), k[0], v / L2[(k[1], k[2])]))

L3_confidence = sorted(L3_confidence, key = lambda x: (-x[2], x[0]))

print("----------(e)----------")

for i in range(5):
  print("{} => {}, confidence is {}".format(L3_confidence[i][0], L3_confidence[i][1], L3_confidence[i][2]))
