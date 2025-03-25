import os
import numpy as np
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from openai import OpenAI

from dotenv import load_dotenv

from core.s3_client import S3FileManager
from core.chunking import markdown_chunking, semantic_chunking, sliding_window_chunking


load_dotenv()
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# DOCUMENTS_KEY = "manual_store.json"
DOCUMENTS_KEY_PKL = "manual_store.pkl"


def save_to_s3_pickle(s3_obj, vectors, key=DOCUMENTS_KEY_PKL):
    try:
        save_file_path = f"{s3_obj.base_path}/{key}"
        pickle_data = pickle.dumps(vectors)
        s3_obj.upload_file(AWS_BUCKET_NAME, save_file_path, pickle_data)
        print(f"Successfully saved pickle to {save_file_path}")
    except Exception as e:
        print(f"Error saving pickle to S3: {str(e)}")

def load_from_s3_pickle(s3_obj, key=DOCUMENTS_KEY_PKL):
    try:
        save_file_path = f"{s3_obj.base_path}/{key}"
        pickle_data = s3_obj.load_s3_pdf(save_file_path)
        return pickle.loads(pickle_data)
    except Exception as e:
        print(f"Error loading pickle from S3: {str(e)}")
        # Return empty list if file doesn't exist
        return []


def read_markdown_file(file, s3_obj):
    content = s3_obj.load_s3_file_content(file)
    return content

def get_embedding(chunks):
    # Ensure chunks is a list of strings
    if isinstance(chunks, str):
        chunks = [chunks]
    
    # Convert any non-string elements to strings
    chunks = [str(chunk) for chunk in chunks]
    
    # Call the OpenAI API
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=chunks
    )
    return response.data

#### BEGIN DOC ####

def get_manual_vector_doc(file, chunks, chunk_strategy, parser):
    vectors = []
    
    # Ensure chunks is a list
    if isinstance(chunks, str):
        chunks = [chunks]
    
    embeddings_data = get_embedding(chunks)
    for i, embed in enumerate(embeddings_data):
        vectors.append({
            "id": f"{file}_{parser}_{chunk_strategy}_chunk_{i}",
            "embedding": list(embed.embedding),
            "metadata": {
                "parser": parser,
                "chunk_type": chunk_strategy,
                "text": chunks[i]
            }
        })
    return vectors


def create_manual_vector_store_doc(file, chunks, chunk_strategy, parser):
    file_name = file.split('/')[2]
    base_path = "/".join(file.split('/')[:-1])
    s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)

    vector = get_manual_vector_doc(file_name, chunks, chunk_strategy, parser)
    save_to_s3_pickle(s3_obj, vector)

    # all_vectors = []
    # content = read_markdown_file(file_name, s3_obj)

    # chunks = markdown_chunking(content, heading_level=2)
    # print(f"markdown Chunk size: {len(chunks)}")
    # vector = get_manual_vector_doc(file, chunks, "markdown", parser)
    # all_vectors.extend(vector)

    # chunks = semantic_chunking(content)
    # print(f"semantic Chunk size: {len(chunks)}")
    # vector = get_manual_vector_doc(file, chunks, "semantic", parser)
    # all_vectors.extend(vector)

    # chunks = sliding_window_chunking(content)
    # print(f"sliding Chunk size: {len(chunks)}")
    # vector = get_manual_vector_doc(file, chunks, "sliding", parser)
    # all_vectors.extend(vector)

    # save_to_s3_pickle(s3_obj, all_vectors)

    return s3_obj

#### END DOC ####


def generate_response_manual(s3_obj, parser, chunking_strategy, query, top_k=5, year=None, quarter=None):
    try:
        query_embedding = get_embedding(query)
        print("Generating manual responses")
        
        # Check if we need to look in mistral directory
        if year and quarter:
            # First try to process direct files from mistral
            md_content = check_mistral_directory(s3_obj, year, quarter[0] if isinstance(quarter, list) else quarter)
            if md_content:
                print(f"Found mistral processed content for {year} {quarter}")
                # Process this content
                chunks = []
                if chunking_strategy == "markdown":
                    chunks = markdown_chunking(md_content, heading_level=2)
                elif chunking_strategy == "semantic":
                    chunks = semantic_chunking(md_content)
                else:
                    chunks = sliding_window_chunking(md_content)
                    
                # If we got chunks, return the most relevant ones
                if chunks:
                    # Get query embedding again (since shape could be different)
                    query_embedding = get_embedding(query)[0].embedding
                    # Get embeddings for all chunks
                    chunk_embeddings_data = get_embedding(chunks)
                    chunk_embeddings = [data.embedding for data in chunk_embeddings_data]
                    
                    # Calculate similarities
                    similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
                    
                    # Get top results
                    top_indices = np.argsort(similarities)[-top_k:][::-1]
                    return [chunks[i] for i in top_indices]
        
        # If above check failed or no mistral content, use original method
        try:
            documents = load_from_s3_pickle(s3_obj)
        except Exception as e:
            print(f"Error loading pickle: {str(e)}")
            # If the manual store doesn't exist yet, create an empty list
            return []
            
        if year and quarter:
            filtered_docs = [doc for doc in documents if 
                                (doc['metadata'].get('year') == year) and 
                                (doc['metadata'].get('quarter') in quarter) and 
                                (doc['metadata'].get('chunk_type') == chunking_strategy) and 
                                (doc['metadata'].get('parser') == parser)]
        else:
            filtered_docs = [doc for doc in documents if 
                                (doc['metadata'].get('chunk_type') == chunking_strategy) and 
                                (doc['metadata'].get('parser') == parser)]
        if not filtered_docs:
            return []

        query_embedding = query_embedding[0].embedding
        doc_embeddings = [doc['embedding'] for doc in filtered_docs]

        similarities = cosine_similarity([query_embedding], doc_embeddings)[0]
        
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        top_docs = [filtered_docs[i] for i in top_indices]
        # top_similarities = [float(similarities[i]) for i in top_indices]

        results = []
        for doc in top_docs:
            results.append(doc['metadata']['text'])

        return results
    except Exception as e:
        print(f"Error in generate_response_manual: {str(e)}")
        # Return empty result instead of raising an error
        return []

def check_mistral_directory(s3_obj, year, quarter):
    """
    Check if there are markdown files in the mistral directory for a specific year and quarter
    
    Args:
        s3_obj: S3FileManager object
        year: Year to check
        quarter: Quarter to check
    
    Returns:
        str: Markdown content if found, None otherwise
    """
    try:
        # Mistral path format would be: mistral/NVIDIA_2022_Q2/extracted_data.md
        mistral_path = f"mistral/NVIDIA_{year}_{quarter}/extracted_data.md"
        print(f"Checking mistral path: {s3_obj.base_path}/{mistral_path}")
        
        try:
            content = s3_obj.load_s3_file_content(mistral_path)
            print(f"Found mistral processed content: {len(content)} bytes")
            return content
        except Exception as e:
            print(f"No content found at mistral path: {str(e)}")
            return None
    except Exception as e:
        print(f"Error checking mistral directory: {str(e)}")
        return None

#### BEGIN NVDIA ####

def get_manual_vector_store(file, chunks, chunk_strategy):
    vectors = []
    file = file.split('/')
    parser = file[1]
    identifier = file[2]
    year = identifier[2:6]
    quarter = identifier[6:]

    embeddings_data = get_embedding(chunks)
    for i, embed in enumerate(embeddings_data):
        vectors.append({
            "id": f"{identifier}_{parser}_{chunk_strategy}_chunk_{i}",
            "embedding": list(embed.embedding),
            "metadata": {
                "year": year,
                "quarter": quarter,
                "parser": parser,
                "chunk_type": chunk_strategy,
                "text": chunks[i]
            }
        })
    return vectors

def create_manual_vector_store():
    base_path = "nvdia/"
    s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)

    files = list({file for file in s3_obj.list_files() if file.endswith('.md')})
    print(files)
    all_vectors = []
    for i, file in enumerate(files):
        print(f"Processing File {i+1}: {file}")
        content = read_markdown_file(file, s3_obj)

        chunks = markdown_chunking(content, heading_level=2)
        print(f"Markdown Chunk size: {len(chunks)}")
        vector = get_manual_vector_store(file, chunks, "markdown")
        all_vectors.extend(vector)

        chunks = semantic_chunking(content)
        print(f"semantic Chunk size: {len(chunks)}")
        vector = get_manual_vector_store(file, chunks, "semantic")
        all_vectors.extend(vector)

        chunks = sliding_window_chunking(content)
        print(f"sliding Chunk size: {len(chunks)}")
        vector = get_manual_vector_store(file, chunks, "sliding_window")
        all_vectors.extend(vector)

    save_to_s3_pickle(s3_obj, all_vectors)

#### END NVDIA ####
