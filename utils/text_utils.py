from string import ascii_lowercase
import re
from bs4 import BeautifulSoup
import pickle
from nltk.corpus import stopwords

replace_dict = {
    'à' : 'a', 'á': 'a', 'é' : 'e', 'è' : 'e', 'ä': 'a', 'ö' : 'o', 'ó' : 'o',
    'í' : 'i', 'ü': 'u', 'ñ' : 'n', 'ú' : 'u', 'ù': 'u', 'ň' : 'n', 'à' : 'a',
    '<3' : '',
}

with open('./utils/contractions.pickle', 'rb') as f:
    contractions = pickle.load(f)

stop_words = set(stopwords.words('english'))

def replace_contractions(text):
    new_text = []
    for word in text.split():
        if word in contractions:
            new_text.append(contractions[word])
        else:
            new_text.append(word)
    return " ".join(new_text)

def clean_text(text):
    text = BeautifulSoup(text, "lxml").text
    text = re.sub(r'&amp;', '', text)
    text = re.sub(r'[_"\-;%()|+&=*%.,!?:#$@\[\]/~`\\\\<\>^\}\{👍👎😝😋😊😄😍♣¡！♠♥★™¿″…]', ' ', text)
    text = re.sub('[m]{2,}', 'mm', text)
    text = re.sub('[o]{2,}', 'oo', text)
    text = re.sub(r'\'', ' ', text)
    text = re.sub(' +', ' ', text)
    for a in list(ascii_lowercase):
        text = re.sub(f'[{a}]{{3,}}', a, text)
    for prior, after in replace_dict.items():
        text = text.replace(prior, after)

    if len(text)==0:
        return text
    if text[0] == " ":
        text = text[1:]
    if len(text)==0:
        return text
    if text[-1] == " ":
        text = text[:-1]

    if len(re.sub(' ', '', text)) == 0:
        return ""
    if  len(re.sub('[0-9a-zA-Z\s]','', text))!=0:
        return ""

    return text

def count_words(count_dict, text):
    for sentence in text:
        for word in sentence.split():
            if word not in count_dict:
                count_dict[word] = 1
            else:
                count_dict[word] += 1


def remove_stopwords(text):
    tokens = [w for w in text.split() if not w in stop_words]
    long_words = []
    for i in tokens:
        if len(i)>1:
            long_words.append(i)
    return (" ".join(long_words)).strip()
    