from __future__ import absolute_import, division, print_function

import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count

from nltk.tokenize.punkt import PunktSentenceTokenizer
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import pandas as pd
from utils import *
import numpy as np
import logging

logger = logging.getLogger(__name__)

# Initalize the path to the saved fine-tuned finBERT model
cl_path = "/Users/julialorenc/Desktop/BAN443_LLMs/FINAL_PROJECT/Fine_tuning/finBERT/models/classifier_model/finbert-sentiment"
model = model = AutoModelForSequenceClassification.from_pretrained(cl_path, cache_dir=None, num_labels=3)
tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased') 

districts = {"at": "Atlanta",
             "bo": "Boston",
             "ch": "Chicago",
             "cl": "Cleveland",
             "da": "Dallas",
             "kc": "Kansas City",
             "mi": "Minneapolis",
             "ny": "New York",
             "ph": "Philadelphia",
             "ri": "Richmond",
             "sf": "San Francisco",
             "sl": "St. Louis"}

def predict(text, model, write_to_csv=False, path=None, use_gpu=False, gpu_name='cuda:0', batch_size=1):
    # Set the model to evaluation mode to disable training-specific operations like dropout
    model.eval()
    
    sentence_tokenizer = PunktSentenceTokenizer()
    sentences = sentence_tokenizer.tokenize(text)

    # Determine the computation device: GPU if available and enabled, otherwise CPU
    device = gpu_name if use_gpu and torch.cuda.is_available() else "cpu"
    logging.info("Using device: %s " % device)
    
    label_list = ['positive', 'negative', 'neutral']
    sentence_scores = [] # Initialize a list to store sentiment scores for each sentence
    
    # Process the sentences in batches
    for batch in chunks(sentences, batch_size):
        # Create InputExample objects for each sentence in the batch
        examples = [InputExample(str(i), sentence) for i, sentence in enumerate(batch)]

        # Convert examples to features compatible with the model
        features = convert_examples_to_features(examples, label_list, 64, tokenizer)

        # Convert features to tensors and move them to the appropriate device
        all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long).to(device)
        all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long).to(device)
        all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long).to(device)

        # Disable gradient calculations for inference
        with torch.no_grad():
            model = model.to(device) # Move the model to the selected device

            # Perform a forward pass to get the logits (raw prediction scores)
            logits = model(all_input_ids, all_attention_mask, all_token_type_ids)[0]
            logging.info(logits)
            # Apply softmax to convert logits to probabilities
            logits = softmax(np.array(logits.cpu()))
            
            # Map predictions to sentiment labels (1 for positive, -1 for negative, 0 for neutral)
            sentiment_mapped = {0: 1, 1: -1, 2: 0}
            predictions = np.squeeze(np.argmax(logits, axis=1))
            
            sentence_scores.append(sentiment_mapped[int(predictions)])
    
    # Calculate the overall tone score as the mean of sentence scores, default to 0 if no scores
    tone_score = np.mean(sentence_scores) if sentence_scores else 0
    
    return tone_score

def process_chunk(chunk):
    chunk["tone"] = chunk.apply(lambda row: predict(row["report"], model), axis=1) # Predict the sentiment of each report
    return chunk

def main():
    reports = pd.read_csv("/Users/julialorenc/Desktop/BAN443_LLMs/FINAL_PROJECT/BIG_Beige_Book_RE.csv")
    # Preprocess and clean the reports
    reports.drop(columns='Unnamed: 0', inplace = True)
    reports["report"] = reports["report"].str.replace("Back to Archive", "")
    reports["report"] = reports["report"].str.replace("Search", "")
    reports["report"] = reports["report"].str.replace("‹ ", "")
    reports['district'] = reports['district'].map(districts)

    chunk_size = 50 # Number of reports to process in parallel
    chunks = np.array_split(reports, len(reports) // chunk_size) # Split the reports into chunks

    num_cores = min(cpu_count(), len(chunks)) # Number of cores to use for parallel processing

    # Activation of the parallel processing
    with Pool(num_cores) as pool: 
        results = pool.map(process_chunk, chunks)

    final_reports = pd.concat(results, ignore_index=True) # Concatenate the processed chunks

    final_reports.to_csv("processed_reports.csv", index=False)
    print("Processing completed. Results saved to 'processed_reports.csv'.")

if __name__ == "__main__":
    main()
