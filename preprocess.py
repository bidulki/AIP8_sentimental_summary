import argparse
import pandas as pd
from tqdm import tqdm
from utils.text_utils import replace_contractions, clean_text, count_words, remove_stopwords

parser = argparse.ArgumentParser()

parser.add_argument('--input_path', required=True, help="raw data filepath")
parser.add_argument('--input_encoding', required=False, default="cp949", help="input file's encoding")
parser.add_argument('--output_path', required=True, help="save filepath")
parser.add_argument('--output_encoding', required=False, default="UTF-8", help="output file's encoding")

args = parser.parse_args()

input_path = args.input_path
output_path = args.output_path

if input_path.split('.')[-1] == "tsv":
    reviews = pd.read_csv(input_path, sep="\t", encoding=args.input_encoding)
else:
    reviews = pd.read_csv(input_path, encoding=args.input_encoding)

reviews = reviews.dropna()
reviews = reviews[["Text", "Summary", "Score"]]
reviews = reviews.drop_duplicates()

print("Text Cleaning...")
clean_texts = []
for text in tqdm(reviews["Text"]):
    text = text.lower()
    text = replace_contractions(text)
    text = clean_text(text)
    text = remove_stopwords(text)
    clean_texts.append(text)

print("Summary Cleaning...")
clean_summaries = []
for summary in tqdm(reviews['Summary']):
    summary = summary.lower()
    summary = replace_contractions(summary)
    summary = clean_text(summary)
    clean_summaries.append(summary)

reviews['Text'] = clean_texts
reviews['Summary'] = clean_summaries

reviews = reviews.drop(reviews[reviews['Summary']==""].index)
reviews = reviews.drop(reviews[reviews['Text']==""].index)
reviews = reviews.drop_duplicates()

text_counts = {}
summary_counts = {}
count_words(text_counts, reviews['Text'])
count_words(summary_counts, reviews['Summary'])
print(f"Size of Text's Vocabulary: {len(text_counts)}")
print(f"Size of Summary's Vocabulary: {len(summary_counts)}")

if output_path.split('.')[-1] == "tsv":
    reviews.to_csv(output_path, sep="\t", index=False, encoding=args.output_encoding)
else:
    reviews.to_csv(output_path, index=False, encoding=args.output_encoding)






    
    