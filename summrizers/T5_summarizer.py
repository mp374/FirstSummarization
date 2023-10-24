# Install the Transformers library

# Import necessary modules
from transformers import *
from transformers import T5ForConditionalGeneration, T5Tokenizer

# Load the T5 model and tokenizer
from transformers.models.swiftformer.convert_swiftformer_original_to_hf import device

model = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

# Define the input text and the summary length
text = "Chaotic and unorganised. Lack of communication with the patient. Rather than that staff and reception " \
             "prefer to walk around, print and have a private chat where you are not included.  After a scan they " \
             "have asked to wait on the corridor when someone will be available to discuss your findings. It was " \
             "after 5.00 pm so everyone went home passing me by like I never existed. I needed to request a Nurse to " \
             "see as I had enough of being ignored. I have been told I am unwell and I have a temperature so maybe " \
             "perhaps I should see a doctor. But they don't have a clue when it will be possible, as he is in the " \
             "ward somewhere. I said definitely no as I have been there since 2.00 pm and any doctors never " \
             "acknowledged me, despite the fact that I was seating there unwell for last three hours waiting for my " \
             "scan and I was passed by at least three doctors where no one said anything.  Really bad service and " \
             "experience. Maybe for the future a patient should be a priority not printing and conversation? "
max_length = 5

# Preprocess the text and encode it as input for the model
input_text = "summarize: " + text
input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

# Generate a summary
summary = model.generate(input_ids, max_length=max_length)

# Decode the summary
summary_text = tokenizer.decode(summary[0], skip_special_tokens=True)

print(summary_text)