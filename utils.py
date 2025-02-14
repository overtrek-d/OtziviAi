import yaml

def get_max():
    with open("texts.yml", 'r') as f:
        text = yaml.safe_load(f)
    texts = []
    for i in text:
        texts.append(len(i[0]))
    print("Max len: ", max(texts))
    return max(texts)

def get_texts():
    with open("texts.yml", 'r') as f:
        text = yaml.safe_load(f)
    texts, labels = [], []
    maxlen = get_max()
    for i in text:
        if len(i[0]) < maxlen:
            i[0] = i[0] + " " * (maxlen - len(i[0]))
        texts.append(i[0].replace(",", " "))
        labels.append(i[1])
    print("texts.yml loaded!")
    return texts, labels
