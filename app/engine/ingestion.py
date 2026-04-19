"""
PDF Document Ingestion Engine for RAG Systems

This module provides an ingestion pipeline that loads PDF documents,
splits them into chunks, generates embeddings using OpenAI, and stores
them in a FAISS vector database for later retrieval.

Usage:
    engine = IngestionEngine(data_path="data/")
    engine.process_documents()

Dependencies:
    - OpenAI API key must be set in .env file
    - FAISS index will be saved to './faiss_index' directory

Author: Abhishek Kavinamoole
Date: 4/18/2026
"""

import os
import logging
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Loading API Keys
load_dotenv()
#Setting up logging
logger = logging.getLogger(__name__)



class IngestionEngine:
     """
     A document ingestion engine that processes PDFs into searchable vector embeddings.
     
     This class handles the complete ETL pipeline for RAG (Retrieval-Augmented Generation):
     1. Loads PDF files from a specified directory
     2. Splits documents into overlapping chunks for context preservation
     3. Generates OpenAI embeddings for each chunk
     4. Builds and saves a FAISS vector index for similarity search
     
     Attributes:
        data_path (str): Path to directory containing PDF files
        embeddings (OpenAIEmbeddings): OpenAI embedding model instance
        text_splitter (RecursiveCharacterTextSplitter): Text splitter with chunk_size=1000, overlap=150
        Example:
            >>> engine = IngestionEngine(data_path="data/")
            >>> engine.process_documents()
            >>> # Vector store saved to 'faiss_index/' for later retrieval
            
     Notes:
        - Requires OPENAI_API_KEY in environment variables
        - Uses text-embedding-3-small model (512 dimensions, cost-efficient)
        - Chunk overlap prevents information loss at boundaries
    """
     def __init__(self, data_path: str):
        """Initialize the IngestionEngine with a data directory and embedding components."""
        self.data_path = data_path
        # Initialize the embeddings model
        self.embeddings = OpenAIEmbeddings(model = "text-embedding-3-small")
        # Initialize the splitter
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = 1000,
            chunk_overlap = 150
        )
        
     def process_documents(self):
        """Execute the complete document ingestion pipeline."""
        if not os.path.exists(self.data_path):
            logger.error(f"Error: {self.data_path} folder not found")
            return
        
        logger.info(f"Loading PDFs from {self.data_path}...")
        try:
            loader = DirectoryLoader(self.data_path, glob = './*.pdf', loader_cls = PyPDFLoader)
            documents = loader.load()

            if not documents:
                logger.warning("No documents found in the specified folder.")
                return
        
            logger.info(f"Successfully loaded {len(documents)} pages.")

            logger.info(f"Splitting {len(documents)} pages into chunks")
            chunks = self.text_splitter.split_documents(documents)
            logger.info(f"Generated {len(chunks)} chunks from the documents.")

            logger.info(f"Generating embeddings for {len(chunks)} chunks using OpenAI")
            vector_store = FAISS.from_documents(chunks, self.embeddings)

            # Save the vector store to disk
            vector_store.save_local("faiss_index")
            logger.info("Ingestion completed and vector store saved to 'faiss_index' folder.")
        
        except Exception as e:
            logger.error(f"An error occurred during ingestion: {str(e)}", exc_info=True)

if __name__ == "__main__":
    """
    Command-line execution block.
    
    This allows the script to be run directly:
        python ingestion_engine.py
    
    Creates an engine instance and processes documents from the 'data/' folder.
    """
    engine = IngestionEngine(data_path = "data/")
    engine.process_documents()
