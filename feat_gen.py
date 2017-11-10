#!/bin/python
import nltk
import os
# import string
from collections import defaultdict

dictionary = {}

def preprocess_corpus(train_sents):
    """Use the sentences to do whatever preprocessing you think is suitable,
    such as counts, keeping track of rare features/words to remove, matches to lexicons,
    loading files, and so on. Avoid doing any of this in token2features, since
    that will be called on every token of every sentence.

    Of course, this is an optional function.

    Note that you can also call token2features here to aggregate feature counts, etc.
    """

    # Loading lexicon
    mypath = os.path.dirname(os.path.realpath(__file__)) + os.path.sep + "data" + os.path.sep + "lexicon"

    global dictionary
    global dictionary_lower
    filelist = {'Geo': ["location"], 
                'TVshow': ["tv.tv_program"],
                'Sports': ["sports.sports_team"],
                'FirstName': ["firstname.5k"],
                'LastName': ["lastname.5000"],
                'Facility': ["architecture.museum"],
                'Stop': ["english.stop"]}
    for label, filenames in filelist.items():
        for filename in filenames:
            f = open(mypath + os.path.sep + filename, "r")
            for line in f.readlines():
                entity_line = line.strip().lower()
                if entity_line == "":
                    continue
                for entity in entity_line.split():
                    # for p in string.punctuation:
                    #     entity = entity.strip(p)
                    if entity in dictionary:
                        if label not in dictionary[entity]:
                            dictionary[entity].append(label)
                    else:
                        dictionary[entity] = [label]

                    entity = entity.lower()

            f.close()

    print "Preprocess finished"


def token2features(sent, i, add_neighs = True):
    """Compute the features of a token.

    All the features are boolean, i.e. they appear or they do not. For the token,
    you have to return a set of strings that represent the features that *fire*
    for the token. See the code below.

    The token is at position i, and the rest of the sentence is provided as well.
    Try to make this efficient, since it is called on every token.

    One thing to note is that it is only called once per token, i.e. we do not call
    this function in the inner loops of training. So if your training is slow, it's
    not because of how long it's taking to run this code. That said, if your number
    of features is quite large, that will cause slowdowns for sure.

    add_neighs is a parameter that allows us to use this function itself in order to
    recursively add the same features, as computed for the neighbors. Of course, we do
    not want to recurse on the neighbors again, and then it is set to False (see code).
    """
    global dictionary

    ftrs = []

    # bias
    ftrs.append("BIAS")

    if i == 0:
        ftrs.append("SENT_BEGIN")
    if i == len(sent)-1:
        ftrs.append("SENT_END")


    if sent[i].startswith("@") or sent[i].startswith("#"):
        tmp = sent[i][1:].lower()
    else:
        tmp = sent[i].lower()
    if tmp in dictionary:
        for dict_name in dictionary[tmp]:
            ftrs.append("IS_" + dict_name)
    # the word itself
    word = unicode(sent[i])


    ftrs.append("WORD=" + word)
    ftrs.append("LCASE=" + word.lower())
    # some features of the word
    if word.isalnum():
        ftrs.append("IS_ALNUM")
    if word.isnumeric():
        ftrs.append("IS_NUMERIC")
    if word.replace(",", "").isdigit():
        ftrs.append("IS_DIGIT")
    if word.isupper():
        ftrs.append("IS_UPPER")
    if word.islower():
        ftrs.append("IS_LOWER")
    # if len(word) > 10:
    #     ftrs.append("LONG_WORD")

    pos = nltk.pos_tag([word])
    if pos:
        ftrs.append("POS_TAG=" + pos[0][1])

    if word.startswith("@"):
        ftrs.append("STARTWITH_@")
    elif word.startswith("#"):
        ftrs.append("STARTWITH_#")
    
    if word.startswith("http"):
        ftrs.append("IS_URL")


    # previous/next word feats
    if add_neighs:
        if i > 0:
            for pf in token2features(sent, i-1, add_neighs = False):
                ftrs.append("PREV_" + pf)
        if i < len(sent)-1:
            for pf in token2features(sent, i+1, add_neighs = False):
                ftrs.append("NEXT_" + pf)

    # return it!
    return ftrs

if __name__ == "__main__":
    sents = [
    # [ "I", "love", "food" ],
    # ["I", "like", "china"]
    ["#Amex", "breach", "by", "Anonymous", "impacts", "77,000", "Cal", "@cardholders"]
    ]
    preprocess_corpus(sents)
    print sents
    for sent in sents:
        for i in xrange(len(sent)):
            print sent[i], ":", token2features(sent, i)
