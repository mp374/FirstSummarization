import torch
from transformers import BartTokenizer, BartForConditionalGeneration

# Load pre-trained BART model and tokenizer
model_name = 'facebook/bart-large-cnn'
tokenizer = BartTokenizer.from_pretrained(model_name)
model = BartForConditionalGeneration.from_pretrained(model_name)

# Example input text for summarization
input_text = "Chaotic and unorganised. Lack of communication with the patient. Rather than that staff and reception " \
             "prefer to walk around, print and have a private chat where you are not included.  After a scan they " \
             "have asked to wait on the corridor when someone will be available to discuss your findings. It was " \
             "after 5.00 pm so everyone went home passing me by like I never existed. I needed to request a Nurse to " \
             "see as I had enough of being ignored. I have been told I am unwell and I have a temperature so maybe " \
             "perhaps I should see a doctor. But they don't have a clue when it will be possible, as he is in the " \
             "ward somewhere. I said definitely no as I have been there since 2.00 pm and any doctors never " \
             "acknowledged me, despite the fact that I was seating there unwell for last three hours waiting for my " \
             "scan and I was passed by at least three doctors where no one said anything.  Really bad service and " \
             "experience. Maybe for the future a patient should be a priority not printing and conversation? "

# Tokenize the input text
inputs = tokenizer(input_text, return_tensors='pt', max_length=1024, truncation=True)

# Generate summary
with torch.no_grad():
    summary_ids = model.generate(inputs['input_ids'], num_beams=4,  max_length=10, min_length=5, early_stopping=True)

# Decode the generated summary
generated_summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)

print("Generated Summary:", generated_summary)