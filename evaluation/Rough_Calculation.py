from rouge_score import rouge_scorer

# List of reference and generated summaries
from summrizers.openAPI import get_summaries

reference_summaries, generated_summaries = get_summaries()

# Initialize ROUGE scorer
scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)

# Calculate ROUGE scores for each pair
all_scores = []
for ref_summary, gen_summary in zip(reference_summaries, generated_summaries):
    scores = scorer.score(ref_summary, gen_summary)
    all_scores.append(scores)

# Calculate average ROUGE scores
avg_rouge1_f = sum(score['rouge1'].fmeasure for score in all_scores) / len(all_scores)
avg_rouge2_f = sum(score['rouge2'].fmeasure for score in all_scores) / len(all_scores)
avg_rougeL_f = sum(score['rougeL'].fmeasure for score in all_scores) / len(all_scores)

# Calculate average ROUGE scores
avg_rouge1_r = sum(score['rouge1'].recall for score in all_scores) / len(all_scores)
avg_rouge2_r = sum(score['rouge2'].recall for score in all_scores) / len(all_scores)
avg_rougeL_r = sum(score['rougeL'].recall for score in all_scores) / len(all_scores)

# Calculate average ROUGE scores
avg_rouge1_p = sum(score['rouge1'].precision for score in all_scores) / len(all_scores)
avg_rouge2_p = sum(score['rouge2'].precision for score in all_scores) / len(all_scores)
avg_rougeL_p = sum(score['rougeL'].precision for score in all_scores) / len(all_scores)


# Print average ROUGE scores
print("Average ROUGE-1 F1:", avg_rouge1_f)
print("Average ROUGE-2 F1:", avg_rouge2_f)
print("Average ROUGE-L F1:", avg_rougeL_f)

print("Average ROUGE-1 R:", avg_rouge1_r)
print("Average ROUGE-2 R:", avg_rouge2_r)
print("Average ROUGE-L R:", avg_rougeL_r)

print("Average ROUGE-1 P:", avg_rouge1_p)
print("Average ROUGE-2 P:", avg_rouge2_p)
print("Average ROUGE-L P:", avg_rougeL_p)