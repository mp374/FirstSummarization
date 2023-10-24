# importing libraries
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize


def sentence_summery(sentence):

    # Tokenizing the text
    stopWords = set(stopwords.words("english"))
    words = word_tokenize(sentence)

    # Creating a frequency table to keep the
    # score of each word


    # need changing to a proper version....
    freqTable = dict()
    for word in words:
        word = word.lower()
        if word in stopWords:
            continue
        if word in freqTable:
            freqTable[word] += 1
        else:
            freqTable[word] = 1

    # Creating a dictionary to keep the score
    # of each sentence
    sentences = sent_tokenize(sentence)
    sentenceValue = dict()

    for sentence in sentences:
        for word, freq in freqTable.items():
            if word in sentence.lower():
                if sentence in sentenceValue:
                    sentenceValue[sentence] += freq
                else:
                    sentenceValue[sentence] = freq

    sentenceAverageValue = dict()
    for sentence, freq in sentenceValue.items():
        sentenceAverageValue[sentence] = sentenceValue[sentence]/len(word_tokenize(sentence))

    sentenceAverageValue = dict(sorted(sentenceAverageValue.items(), key=lambda item: item[1], reverse=True))

    sumValues = 0
    for sentence in sentenceValue:
        sumValues += sentenceValue[sentence]

    # Average value of a sentence from the original text

    average = int(sumValues / len(sentenceValue))

    # Storing sentences into our summary.
    summary = '\nSummery from count\n'
    for sentence in sentences:
        if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
            summary += " " + sentence
    print(summary)


    # Storing sentences into our summary.
    summary = '\nSummery from average count\n'
    for sentence in sentences:
        if sentence in list(sentenceAverageValue.keys())[0:2]:
            summary += " " + sentence
    print(summary)



