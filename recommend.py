import numpy as np
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

# Load the FAISS index and the dataset
index1 = faiss.read_index('index_without_desc')
index2 = faiss.read_index('index')
df = pd.read_csv('books.csv')

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

def construct_query():
    print("Enter the details for the book:")
    title = input("Title: ")
    authors = input("Authors: ")
    categories = input("Categories: ")

    if not any([title, authors, categories]):
        print("You must enter at least one field.")
        return construct_query()

    query_string = f"""Title: {title or ''}\nAuthors: {authors or ''}\nCategories: {categories or ''}\n"""
    return query_string

def search_books():
    # Step 1: Get user input and perform the initial search
    query_string = construct_query()
    query_embedding = np.array([embedding_model.embed_query(query_string)], dtype='float32')

    D, I = index1.search(query_embedding, k=1)  # Get the top book
    top_book = df.iloc[I.flatten()].iloc[0]

    print("\nTop matched book:")
    print(f"Title: {top_book['title']}")
    print(f"Authors: {top_book['authors']}")
    print(f"Categories: {top_book['categories']}")
    print(f"Description: {top_book['description']}")
    print("\n---\n")

    # Step 2: Construct a query using the entire metadata of the top book
    secondary_query = f"""Title: {top_book['title']}\nAuthors: {top_book['authors']}\nCategories: {top_book['categories']}\nDescription: {top_book['description']}\n"""
    secondary_query_embedding = np.array([embedding_model.embed_query(secondary_query)], dtype='float32')

    # Step 3: Perform a secondary search
    D2, I2 = index2.search(secondary_query_embedding, k=5)

    # Step 4: Retrieve the top 5 results
    best_matches = df.iloc[I2.flatten()]

    print("\nTop 5 similar books:\n")
    for idx, row in best_matches.iterrows():
        print(f"Title: {row['title']}")
        print(f"Authors: {row['authors']}")
        print(f"Categories: {row['categories']}")
        print(f"Description: {row['description']}")
        print("\n---\n")

if __name__ == "__main__":
    search_books()
