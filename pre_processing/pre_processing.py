import re
import string
import emoji
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

# nltk.download("stopwords")
# nltk.download("punkt")
# nltk.download("wordnet")
# nltk.download('omw-1.4')
# nltk.download('words')


import nltk
import time
words = set(nltk.corpus.words.words())

def clean_text(text, tokenizer, stopwords, rootWord):
    """Pre-processes a text and generates tokens.
    Args:
        text: Text to tokenize.
        tokenizer: A tokenizer of your choice.
        stopwords: A list of stop words to be used.
        rootWord: A string specifying the root word finding method.
    Returns:
        tokens: Tokenized and cleaned text.
    """

    text = str(text).lower()  # Lowercase words
    text = re.sub(r"\s+", " ", text)  # Remove multiple spaces in content
    text = re.sub(r"\w+…|…", "", text)  # Remove ellipsis (and last word)
    text = re.sub(r"(?<=\w)-(?=\w)", " ", text)  # Replace dash between words
    text = text.replace("_", " ")  # Replace underscore between words
    text = re.sub(
        f"[{re.escape(string.punctuation)}]", "", text
    )  # Remove punctuation.
    text = re.sub(r'\w*\d\w*', '', text).strip()  # remove words with numbers.

    tokens = tokenizer(text)  # Get tokens from text.
    tokens = [t for t in tokens if not t in stopwords]  # Remove stopwords.
    tokens = ["" if t.isdigit() else t for t in tokens]  # Remove digits.
    tokens = [t for t in tokens if t in words]  # removing non-english words.
    tokens = [t for t in tokens if not any(c in emoji.unicode_codes.EMOJI_DATA for c in t)]  # Remove tokens with emojis.

    if rootWord == "Stem":
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]  # get the stem of the word.
    elif rootWord == "Lemma":
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]  # get the lemma of the word.

    return tokens


def pre_process_tokenize(list_stories, rootWord):
    """Generates a list of pre-processed and tokenized stories.
        Args:
            list_stories: list of original stories from CareOpinion APIs.
        Returns:
            tokenized_stories: A list of tokenized and cleaned stories.
    """

    default_stop_words = stopwords.words("english")
    default_stop_words.extend(['amp', "hca", "epu", "am", "pm", "ah", "ai", "aa", "abu", "abut", "ie"])
    stopwords_for_the_context = set(default_stop_words)
    tokenized_stories = []

    for story in list_stories:
        tokenized_story = clean_text(story, word_tokenize, stopwords_for_the_context, rootWord)
        tokenized_stories.append(tokenized_story)

    return tokenized_stories


def preprocess(list_stories, rootWord):

    default_stop_words = stopwords.words("english")
    default_stop_words.extend(['amp', "hca", "epu", "am", "pm", "ah", "ai", "aa", "abu", "abut", "ie"])
    stopwords_for_the_context = set(default_stop_words)
    tokenized_stories = []

    for story in list_stories:
        tokenized_story = clean_text(story, word_tokenize, stopwords_for_the_context, rootWord)
        tokenized_stories.append(tokenized_story)

    preprocessed_sentences = [' '.join(sentence_tokens) for sentence_tokens in tokenized_stories]

    return preprocessed_sentences