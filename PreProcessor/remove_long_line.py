
path = "C:\work\Code\Community-Sentiment-Analysis\Cluster\Cluster\data\\cluster_6.txt"

lines = open(path).readlines()

for i, line in enumerate(lines):
    if len(line) > 1000:
        print i, len(line)