from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceBgeEmbeddings

class RetrievalService:
    def __init__(self, db_path="chroma_db", collection_name="constitution_db"):
        embedding_model = HuggingFaceBgeEmbeddings(
            model_name="BAAI/bge-large-en-v1.5",
            encode_kwargs={"normalize_embeddings": True}
        )
        self.vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embedding_model,
            persist_directory=db_path
        )

    def retrieve_context(self, query, top_k=5):
        results = self.vectorstore.similarity_search(query, k=top_k)
        context = " ".join([result.page_content for result in results])
        return context