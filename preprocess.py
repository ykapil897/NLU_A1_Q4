import string

def tokenize(text):
    text = text.lower()
    for p in string.punctuation:
        text = text.replace(p, " ")
    return text.split()
