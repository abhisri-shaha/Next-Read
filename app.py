from flask import Flask, request, render_template
import numpy as np
import faiss
from langchain.embeddings import HuggingFaceEmbeddings
import pandas as pd

app = Flask(__name__)

# Load the FAISS indexes and dataset
index1 = faiss.read_index('index_without_desc')
index2 = faiss.read_index('index')
df = pd.read_csv('books.csv')

# Load the embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

@app.route('/')
def home():
    return render_template('index.html')  # Render a simple form for user input

@app.route('/recommend', methods=['POST'])
def recommend():
    # Step 1: Get user input
    title = request.form.get('title', '').strip()
    authors = request.form.get('authors', '').strip()
    categories = request.form.get('categories', '').strip()

    if not any([title, authors, categories]):
        return "You must enter at least one field.", 400

    query_string = f"""Title: {title}\nAuthors: {authors}\nCategories: {categories}\n"""

    # Step 2: Perform the initial search
    query_embedding = np.array([embedding_model.embed_query(query_string)], dtype='float32')
    D, I = index1.search(query_embedding, k=1)

    # Retrieve the metadata of the top book
    top_book = df.iloc[I.flatten()].iloc[0]

    # Step 3: Construct a query using the entire metadata of the top book
    secondary_query = f"""Title: {top_book['title']}\nAuthors: {top_book['authors']}\nCategories: {top_book['categories']}\nDescription: {top_book['description']}\n"""
    secondary_query_embedding = np.array([embedding_model.embed_query(secondary_query)], dtype='float32')

    # Step 4: Perform a secondary search to get top 15 books
    D2, I2 = index2.search(secondary_query_embedding, k=15)

    # Retrieve the top 15 results
    best_matches = df.iloc[I2.flatten()]

    # Render results
    return render_template('results.html', books=best_matches.to_dict(orient='records'))

if __name__ == "__main__":
    app.run(debug=True)
