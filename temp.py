import pandas as pd
from collections import Counter
file = 'prediction.txt'
df = pd.read_csv(file)
l = list(df.iloc[:,2])
cnt = Counter()
for word in l:
    cnt[word] += 1

out = sorted(cnt, key=cnt.get, reverse=True)