import re
import pandas as pd


# get the useless junk out of the text and make it all lowercase
# all we want are words
def clean_text(string):
    stopwords = gather_stopwords()
    string = re.sub(" +", ",", re.sub("(\< ?(br|i|/i|em|spoiler|hr) ?\/?\>)|[^a-zA-Z ]", "", string)).lower()
    string = string.split(",")
    string = [x for x in string if x not in stopwords]
    output = ""
    for x in string:
        output += x + " "
    return output


# read stopwords and return array of words
def gather_stopwords():
    f = open("stopwords.en.txt", "r")
    words = []
    for x in f:
        if x != "":
            words.append(re.sub("\s", "", x))
    return words


# get the reviews, clean them up, and save them to a csv
train = pd.read_csv("imdb_te.csv", encoding = "ISO-8859-1")
train.to_csv('test.csv', index=False)