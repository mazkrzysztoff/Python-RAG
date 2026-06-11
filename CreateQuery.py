from langchain_community.chat_models.openai import ChatOpenAI
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from langchain_core.chat_history import InMemoryChatMessageHistory

API_KEY = ""
PROMPT_TEMPLATE = """ 
Answer based only on the following information:

{context}

---

Answer the question based on the above context: {question}
"""



def main() :
    embedding_function = OpenAIEmbeddings(api_key= API_KEY)

    client = QdrantClient(url="http://localhost:6333")

    db = QdrantVectorStore(
        client=client,
        collection_name="documents",
        embedding=embedding_function,
    )

    query = "How does Alice meet the Mad Hatter?"

    results = db.similarity_search_with_relevance_scores(query, k = 3)
    if(len(results)) == 0 :
        print("Found no results")
        return
    text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])

    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context = text, question = query)


    history = InMemoryChatMessageHistory()

    model = ChatOpenAI(model = "gpt-5.2",api_key= API_KEY)


    while True :
        history.add_user_message(prompt)
        responce = model.invoke(history.messages)
        history.add_ai_message(responce)

        print(responce)

        prompt = input(" ")

if __name__ == "__main__" :
    main()
