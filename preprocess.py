import spacy
from collections import namedtuple
import string

nlp = spacy.load("en_core_web_sm", disable=["tagger", "parser"])

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
        "n’t",
    ]
)

# name of all features
feat_cols = [
    "w_overlap",
    "w_hyp_extra",
    "w_jaccard",
    "w_jaccard_s",
    "ne_overlap",
    "ne_hyp_extra",
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
    t_negations, text_proc, text_ne = preprocess_sentence(text)
    h_negations, hyp_proc, hyp_ne = preprocess_sentence(hyp)

    # get features by using preprocessed sentences
    word_pair = Pair(text_proc, hyp_proc)
    w_overlap = overlap(word_pair)
    w_hyp_extra = hyp_extra(word_pair)

    # jaccard similarity between the text and h
    w_jaccard = w_overlap / (len(word_pair.t) + len(word_pair.h))

    # min sim
    w_jaccard_s = w_overlap / min((len(word_pair.t), len(word_pair.h)))

    # get features by using the named entity recognized
    ne_pair = Pair(text_ne, hyp_ne)
    ne_overlap = overlap(ne_pair)
    ne_hyp_extra = hyp_extra(ne_pair)

    return (
        w_overlap,
        w_hyp_extra,
        w_jaccard,
        w_jaccard_s,
        ne_overlap,
        ne_hyp_extra,
        t_negations,
        h_negations,
    )

