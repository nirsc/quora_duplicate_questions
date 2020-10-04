from utils.embeddings import *
import pandas as pd


DATA_DIR = '..//data//'
filename = 'quora_duplicate_questions.tsv'
df = pd.read_csv(DATA_DIR+filename,delimiter='\t')
print(df.head())
