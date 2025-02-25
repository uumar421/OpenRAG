import os
import argparse
import pytesseract
from pdf2image import convert_from_path
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.vectorstores import Chroma
from langchain.schema import Document

class PDFProcessor:
    def __init__(self, chroma_db_path="chroma_db", collection_name="vector_db"):
        self.chroma_db_path = chroma_db_path
        self.collection_name = collection_name
        self.embedding_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5", encode_kwargs={"normalize_embeddings": True}
        )

    def extract_text(self, pdf_path):
        images = convert_from_path(pdf_path)
        page_texts = [pytesseract.image_to_string(image) for image in images]
        print(f"Extracted text from {len(images)} pages.")

        return "\n\n".join(page_texts)

    def split_text(self, text, chunk_size=1000, chunk_overlap=200):
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_text(text)
        print(f"Split text into {len(chunks)} chunks.")

        return chunks

    def get_document_id(self):
        if not os.path.exists(self.chroma_db_path):
            return 1 

        vectorstore = Chroma(persist_directory=self.chroma_db_path, collection_name=self.collection_name)
        existing_docs = vectorstore.get(include=['metadatas'])
        doc_ids = [meta['id'] for meta in existing_docs['metadatas'] if 'id' in meta]

        return max(doc_ids) + 1 if doc_ids else 1

    def store_embeddings(self, chunks, document_id, document_name):
        documents = [
            Document(page_content=chunk, metadata={"id": document_id, "name": document_name})
            for chunk in chunks
        ]
        vectorstore = Chroma.from_documents(
            documents=documents, embedding=self.embedding_model,
            collection_name=self.collection_name, persist_directory=self.chroma_db_path
        )
        print(f"Stored {len(chunks)} chunks in ChromaDB.")

    def process_pdf(self, pdf_path):
        print("Processing PDF...")
        text = self.extract_text(pdf_path)
        chunks = self.split_text(text)
        document_id = self.get_document_id()
        document_name = os.path.splitext(os.path.basename(pdf_path))[0]
        
        print(f"Using document ID: {document_id}, Document Name: {document_name}")
        self.store_embeddings(chunks, document_id, document_name)
        print("Processing complete.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Process a PDF and store embeddings in ChromaDB.")
    parser.add_argument("pdf_path", type=str, help="Path to the PDF document")
    args = parser.parse_args()

    processor = PDFProcessor()
    processor.process_pdf(args.pdf_path)