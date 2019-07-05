import en_core_web_sm
import string
from collections import namedtuple

nlp = en_core_web_sm.load(disable=["tagger", "parser"])


Pair = namedtuple("Pair", ["t", "h"])

# Store list of punctuation marks to ignore
PUNCT = string.punctuation
NEGATIONS = set(
    [
        "deny",
        "fail",
        "never",
        "no",
        "nobody",
        "not",
        "nothing",
        "reject",
        "without",
        "noone",
        "none",
        "cannot",
        "nor",
        "n‘t",
        "n't",
    ]
)

# name of all features
feat_cols = [
    "w_overlap",
    "w_hyp_extra",
    "w_txt_extra",
    "w_jaccard",
    "w_jaccard_s",
    "ne_overlap",
    "ne_hyp_extra",
    "ne_txt_extra",
    "t_negations",
    "h_negations",
]


def preprocess_sentence(sentence):
    """
    Preprocess a sentence by removing stop words and punctuations, and lemmatizing
    Args: 
        sentence (string): full sentence string
    Returns:
        processed_sentence (list): list of processed tokens
    """

    doc = nlp(sentence)
    processed_sentence = []
    named_entities = []

    has_negation = 0

    # filtering stop words and lemmatizing
    for word in doc:
        if word.is_stop == False and word.text not in PUNCT:
            processed_sentence.append(word.lemma_.lower().strip())
        if word.text in NEGATIONS:
            has_negation += 1

    # Named entitities recognized
    for word in doc.ents:
        named_entities.append(word.text.strip())

    return has_negation, processed_sentence, named_entities


def overlap(pair):
    """
    Compute the overlap material
    """
    return len(set(pair.t).intersection(pair.h))


def hyp_extra(pair):
    """
    Compute the number of words occuring in hyp but not in text
    """
    return len(set(pair.h) - set(pair.t))


def preprocess_pair(text, hyp):
    """
    Preprocess pair of text and hypothesis
    Args:
        text (string): text sentence 
        hyp (string): hypothesis sentence to compare to text
    Returns:
        features (tuple): tuple of features that can be used
            for predictions
    """
    # process the text and hyp sententences
    t_negations, txt_proc, txt_ne = preprocess_sentence(text)
    h_negations, hyp_proc, hyp_ne = preprocess_sentence(hyp)

    # get features by using preprocessed sentences
    word_pair = Pair(txt_proc, hyp_proc)
    w_overlap = overlap(word_pair)
    w_hyp_extra = len(set(word_pair.h) - set(word_pair.t))
    w_txt_extra = len(set(word_pair.t) - set(word_pair.h))

    # jaccard similarity between the text and h
    w_jaccard = w_overlap / max((len(word_pair.t) + len(word_pair.h)), 1)

    # min sim
    w_jaccard_s = w_overlap / max(min((len(word_pair.t), len(word_pair.h))), 1)

    # get features by using the named entity recognized
    ne_pair = Pair(txt_ne, hyp_ne)
    ne_overlap = overlap(ne_pair)
    ne_hyp_extra = len(set(ne_pair.h) - set(ne_pair.t))
    ne_txt_extra = len(set(ne_pair.t) - set(ne_pair.h))

    return (
        w_overlap,
        w_hyp_extra,
        w_txt_extra,
        w_jaccard,
        w_jaccard_s,
        ne_overlap,
        ne_hyp_extra,
        ne_txt_extra,
        t_negations,
        h_negations,
    )

