import html
import re
import string
import unicodedata

from nltk.tokenize import word_tokenize


def remove_special_chars(text):
    re1 = re.compile(r'  +')
    x1 = text.lower().replace('#39;', "'").replace('amp;', '&').replace('#146;', "'").replace(
        'nbsp;', ' ').replace('#36;', '$').replace('\\n', "\n").replace('quot;', "'").replace(
        '<br />', "\n").replace('\\"', '"').replace('<unk>', 'u_n').replace(' @.@ ', '.').replace(
        ' @-@ ', '-').replace('\\', ' \\ ')
    return re1.sub(' ', html.unescape(x1))


def remove_non_ascii(text):
    """Remove non-ASCII characters from list of tokenized words"""
    return unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8', 'ignore')


def to_lowercase(text):
    return text.lower()


def remove_punctuation(text):
    """Remove punctuation from list of tokenized words"""
    translator = str.maketrans('', '', string.punctuation)
    return text.translate(translator)


def replace_numbers(text):
    """Replace all interger occurrences in list of tokenized words with textual representation"""
    text = re.sub(r'\d+(\.\d+)*\sx\s*\d+(\.\d+)*', '1', text)
    text = re.sub(r'\d+\.\d+', '1', text)
    text = re.sub(r't\d+.', 'N ', text)
    text = re.sub(r'l\d+.', 'N ', text)
    text = re.sub(r'\d+\s*th', '1 ', text)
    text = re.sub(r'\d+\s*cm', '1 ', text)
    text = re.sub(r'\d+\s*mm', '1 ', text)
    return re.sub(r'\d+.', 'NUM ', text)


def remove_whitespaces(text):
    return text.strip()


def remove_stopwords(words, stop_words):
    """
    :param words:
    :type words:
    :param stop_words: from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
    or
    from spacy.lang.en.stop_words import STOP_WORDS
    :type stop_words:
    :return:
    :rtype:
    """
    return [word for word in words if word not in stop_words]


def text2words(text):
    return word_tokenize(text)


def normalize_text(text):
    text = remove_special_chars(text)
    text = remove_non_ascii(text)
    text = to_lowercase(text)
    text = replace_numbers(text)

    text = text.replace("dual-xxxx", "xxxx")
    text = text.replace("xxxx-a-xxxx", "xxxx")
    text = text.replace("(xxxx)", "xxxx")
    text = text.replace("x-xxxx", "xxxx")
    text = text.replace("xxxx xxxx", "xxxx").replace("xxxx xxxx", "xxxx")
    text = word_tokenize(text)
    text = ' '.join(text)

    return text