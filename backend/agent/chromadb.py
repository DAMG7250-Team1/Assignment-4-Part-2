import os
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from chromadb.config import Settings
from fastapi import HTTPException

import tempfile
from dotenv import load_dotenv

from core.chunking import markdown_chunking, semantic_chunking, sliding_window_chunking
from core.s3_client import S3FileManager
# from s3 import S3FileManager
# from chunk_strategy import markdown_chunking, semantic_chunking, sliding_window_chunking

load_dotenv()
AWS_BUCKET_NAME = os.getenv("AWS_BUCKET_NAME")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


def read_markdown_file(file, s3_obj):
    content = s3_obj.load_s3_file_content(file)
    return content

def get_chroma_embeddings(texts):
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
                api_key=OPENAI_API_KEY,
                model_name="text-embedding-3-small"
            )
    return openai_ef(texts)


def create_chromadb_vector_store(chroma_client, file, chunks, chunk_strategy):
    # Create collections
    collection_doc_sem = chroma_client.get_or_create_collection(name="docling_semantic")
    collection_mist_sem = chroma_client.get_or_create_collection(name="mistral_semantic")
    collection_doc_mark = chroma_client.get_or_create_collection(name="docling_markdown")
    collection_mist_mark = chroma_client.get_or_create_collection(name="mistral_markdown")
    collection_doc_slid = chroma_client.get_or_create_collection(name="docling_sliding_window")
    collection_mist_slid = chroma_client.get_or_create_collection(name="mistral_sliding_window")

    file = file.split('/')
    parser = file[1]
    identifier = file[2]
    year = identifier[2:6]
    quarter = identifier[6:]

    base_metadata = {
            "year": year,
            "quarter": quarter
        }
    metadata = [base_metadata for _ in range(len(chunks))]
    
    embeddings = get_chroma_embeddings(chunks)
    ids = [f"{parser}_{chunk_strategy}_{identifier}_{i}" for i in range(len(chunks))]
    if parser == 'docling' and chunk_strategy == 'semantic':
        print(f"adding to collection - {parser} - {chunk_strategy}")
        collection_doc_sem.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadata,
                documents=chunks
            )

    elif parser == 'docling' and chunk_strategy == 'markdown':
        print(f"adding to collection - {parser} - {chunk_strategy}")
        collection_doc_mark.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadata,
                documents=chunks
            )

    elif parser == 'mistral' and chunk_strategy == 'semantic':
        print(f"adding to collection - {parser} - {chunk_strategy}")
        collection_mist_sem.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadata,
                documents=chunks
            )

    elif parser == 'mistral' and chunk_strategy == 'markdown':
        print(f"adding to collection - {parser} - {chunk_strategy}")
        collection_mist_mark.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadata,
                documents=chunks
            )
    elif parser == 'docling' and chunk_strategy == 'sliding_window':
        print(f"adding to collection - {parser} - {chunk_strategy}")
        collection_doc_slid.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadata,
                documents=chunks
            )
    elif parser == 'mistral' and chunk_strategy == 'sliding_window':
        print(f"adding to collection - {parser} - {chunk_strategy}")
        collection_mist_slid.upsert(
                ids=ids,
                embeddings=embeddings,
                metadatas=metadata,
                documents=chunks
            )

def upload_directory_to_s3(local_dir, s3_obj, s3_prefix):
    """Upload a directory and its contents to S3"""
    for root, dirs, files in os.walk(local_dir):
        for file in files:
            local_path = os.path.join(root, file)
            # Create the S3 key by replacing the local directory path with the S3 prefix
            relative_path = os.path.relpath(local_path, local_dir)
            s3_key = f"{s3_obj.base_path}/{os.path.join(s3_prefix, relative_path)}".replace("\\", "/")
            with open(local_path, "rb") as f:
                s3_obj.upload_file(AWS_BUCKET_NAME, s3_key, f.read())

###### For querying - Fast API  #######

def download_chromadb_from_s3(s3_obj, temp_dir):
    """Download ChromaDB files from S3 to a temporary directory"""
    s3_prefix = f"{s3_obj.base_path}/chroma_db"
    s3_files = [f for f in s3_obj.list_files() if f.startswith(s3_prefix)]
    
    for s3_file in s3_files:
        # Extract the relative path from the S3 key
        relative_path = s3_file[len(s3_prefix):].lstrip('/')
        local_path = os.path.join(temp_dir, relative_path)
        
        # Create the directory if it doesn't exist
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        # Download the file from S3
        content = s3_obj.load_s3_pdf(s3_file)
        with open(local_path, 'wb') as f:
            f.write(content if isinstance(content, bytes) else content.encode('utf-8'))

def query_chromadb(parser, chunking_strategy, query, top_k, year, quarter):
    with tempfile.TemporaryDirectory() as temp_dir:
        try:
            s3_obj = S3FileManager(AWS_BUCKET_NAME, "nvidia/")
            download_chromadb_from_s3(s3_obj, temp_dir)
            chroma_client = chromadb.PersistentClient(path=temp_dir)
            try:
                collection = chroma_client.get_collection(f"{parser}_{chunking_strategy}")
            except Exception as e:
                raise HTTPException(status_code=404, detail=f"Collection not found: {str(e)}")

            query_embeddings = get_chroma_embeddings([query])
            where_filter = {
                        "$and": [
                            {"quarter": {"$in": quarter}},
                            {"year": {"$eq": year}}
                        ]
                    }
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=top_k,
                where=where_filter
            )       
            return results["documents"]
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error querying ChromaDB: {str(e)}")


# def main():
    # with tempfile.TemporaryDirectory() as temp_dir:
    #     chroma_client = chromadb.PersistentClient(path=temp_dir)
    #     base_path = "nvdia/"
    #     s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)

    #     files = list({file for file in s3_obj.list_files() if file.endswith('.md')})
    #     files = files[:2]  #make change here
    #     for file in files:
    #         content = read_markdown_file(file, s3_obj)

    #         # For markdown chunking strategy
    #         chunks_mark = markdown_chunking(content, heading_level=2)
    #         print(f"Chunk size markdown: {len(chunks_mark)}")
    #         create_chromadb_vector_store(chroma_client, file, chunks_mark, "markdown")

    #         # For semantic chunking strategy
    #         chunks_sem = semantic_chunking(content, max_sentences=10)
    #         print(f"Chunk size semantic: {len(chunks_sem)}")
    #         create_chromadb_vector_store(chroma_client, file, chunks_sem, "semantic")

    #     print(files)

    #     # Upload the entire ChromaDB directory to S3
    #     upload_directory_to_s3(temp_dir, s3_obj, "chroma_db")
        
    #     print("ChromaDB has been uploaded to S3.")
    #     # Explicitly close the ChromaDB client

def create_chromadb():
    with tempfile.TemporaryDirectory() as temp_dir:
        chroma_client = chromadb.PersistentClient(path=temp_dir)
        base_path = "nvdia/"
        s3_obj = S3FileManager(AWS_BUCKET_NAME, base_path)

        files = list({file for file in s3_obj.list_files() if file.endswith('.md')})
        # files = files[:2]  # process only first two files
        for file in files:
            print(file)
            content = read_markdown_file(file, s3_obj)
            print("\nread markdown")
            # For markdown chunking strategy
            chunks_mark = markdown_chunking(content, heading_level=2)
            print(f"Chunk size markdown: {len(chunks_mark)}")
            create_chromadb_vector_store(chroma_client, file, chunks_mark, "markdown")

            # For semantic chunking strategy
            chunks_sem = semantic_chunking(content, max_sentences=10)
            print(f"Chunk size semantic: {len(chunks_sem)}")
            create_chromadb_vector_store(chroma_client, file, chunks_sem, "semantic")

            # For Sliding chunking strategy
            chunks_sliding = sliding_window_chunking(content)
            print(f"Chunk size Sliding: {len(chunks_sem)}")
            create_chromadb_vector_store(chroma_client, file, chunks_sliding, "sliding_window")

        print(files)

        # Upload the entire ChromaDB directory to S3
        upload_directory_to_s3(temp_dir, s3_obj, "chroma_db")
        print("ChromaDB has been uploaded to S3.")

# if __name__ == "__main__":
#     main()