import textstat
from textblob import TextBlob

from summrizers.openAPI import get_summaries


def evaluate_readability(summary):
    readability_score = textstat.flesch_kincaid_grade(summary)

    return readability_score


summary, generated_summaries = get_summaries()
readability_scores = [evaluate_readability(summary) for summary in generated_summaries]

print("Min Readability Score:", min(readability_scores))
print("Max Readability Score:", max(readability_scores))
print("Average Readability Score:", sum(readability_scores)/len(readability_scores))