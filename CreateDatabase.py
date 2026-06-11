from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_qdrant import QdrantVectorStore


from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

DATA_PATH = ""
API_KEY = ""
NAME = "test"

def load_data() :
    loader = DirectoryLoader(DATA_PATH,glob="*.md")
    documents = loader.load()
    return documents

def main():
    splitter = RecursiveCharacterTextSplitter(
        chunk_size= 800,
        chunk_overlap= 100,
        length_function= len,
        add_start_index= True
    )


    documents = load_data()
    chunks = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key = API_KEY,
                                  chunk_size= 20)

    client = QdrantClient(url="http://localhost:6333")

    try:
        client.get_collection(NAME)
        print("Collection already exists.")
    except Exception:
        client.create_collection(
            collection_name=NAME,
            vectors_config=VectorParams(
                size=1536,
                distance=Distance.COSINE,
            ),
        )
        print("Collection created.")

    vectorstore = QdrantVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        url="http://localhost:6333",
        collection_name=NAME,
    )

    vectorstore.add_documents(chunks)


if __name__ == "__main__":
    main()