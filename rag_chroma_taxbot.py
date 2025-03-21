from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from chromadb.utils.embedding_functions import DefaultEmbeddingFunction
import os

# Load the PDF document
def load_pdf(path: str):
    loader = PyPDFLoader(path)
    return loader.load()



# Split documents into smaller chunks
def split_documents(docs, chunk_size=1000, chunk_overlap=200):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    return splitter.split_documents(docs)



# Store the document chunks in ChromaDB using the default embedding function
def store_in_chroma(documents, persist_dir="chroma_store"):
    default_embedder = DefaultEmbeddingFunction()
    vectordb = Chroma.from_documents(
        documents=documents,
        embedding=default_embedder,
        persist_directory=persist_dir
    )
    vectordb.persist()
    print(f"Stored {len(documents)} chunks in ChromaDB.")
    return vectordb



# Query ChromaDB with a user query
def query_chroma(query: str, persist_dir="chroma_store", k=3):
    embedder = DefaultEmbeddingFunction()
    vectordb = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedder
    )
    results = vectordb.similarity_search(query, k=k)
    return results
  

# Print the matched chunks
def print_results(results):
    print("\nTop Matching Chunks\n" + "-" * 30)
    for i, doc in enumerate(results):
        print(f"\nChunk {i + 1}:\n{doc.page_content.strip()}")
      

# Main 
if __name__ == "__main__":
    file_path = "us_tax_laws_2024.pdf"  

    # Load and store if DB doesn't exist
    if not os.path.exists("chroma_store"):
        print("Loading and indexing the PDF...")
        raw_docs = load_pdf(file_path)
        chunks = split_documents(raw_docs)
        store_in_chroma(chunks)
    else:
        print("ChromaDB already exists. Skipping PDF processing.")

    # Query input
    user_query = input("\nAsk a question about US Tax Law 2024: ")
    top_docs = query_chroma(user_query)
    print_results(top_docs)
