import pandas as pd
import numpy as np
import faiss
from langchain.embeddings import HuggingFaceEmbeddings

df = pd.read_csv('books.csv')

def string_representation(row):
    string_rep = f"""Title: {row['title']}
Authors: {row['authors']}
Categories: {row['categories']}
"""

    return string_rep

df['string_representation'] = df.apply(string_representation, axis=1)

embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

dim = 384  # Dimension of "all-MiniLM-L6-v2" embeddings

index = faiss.IndexFlatL2(dim)

X = np.zeros((len(df['string_representation']), dim), dtype='float32')

for i, text in enumerate(df['string_representation']):
    if i % 100 == 0:
        print(f"Processing {i}/{len(df)}")
    embedding = embedding_model.embed_query(text)
    X[i] = np.array(embedding)

index.add(X)

faiss.write_index(index, 'index_without_desc')

index = faiss.read_index('index_without_desc')

fav_book = df.iloc[2723]

# print(fav_book['string_representation'])

embedding = np.array([embedding_model.embed_query(fav_book['string_representation'])], dtype='float32')

D, I = index.search(embedding, k=5)

best_matches = np.array(df['string_representation'])[I.flatten()]

for similar in best_matches:
  print(similar)
  print()