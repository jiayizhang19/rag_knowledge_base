import os
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain

load_dotenv()
hf_token = os.getenv("hf_token")
os.environ['HF_API_TOKEN'] = hf_token
embedding_model = "all-MiniLM-L6-v2"
# llm_model = "mistralai/Mistral-7B-v0.1" # Supported task: Conversational
llm_model = "google/gemma-2-2b-it"


login(token=hf_token)

def run_mvp():
    # 1. Ingestion (Loading your Library)
    print("Step1: Ingesting data ...")
    loader = DirectoryLoader(
        "./mock-uni-data", 
        glob="./*.txt", 
        loader_cls=TextLoader
    )
    raw_docs = loader.load()

    # 2. Chunking (Cutting the Books into Pages)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 600,
        chunk_overlap = 100,
    )
    docs = text_splitter.split_documents(raw_docs)

    # 3. Embedding (Local math)
    print("Step 2: Creating embeddings ...")
    embeddings = HuggingFaceEmbeddings(model_name=embedding_model)

    # 4. Vector store (Save to local memory)
    vector_store = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory="./uni_db"
    )

    # 5. Free LLM (The Brain)
    print("Step 3: Initializing free LLM ...")
    llm = HuggingFaceEndpoint(
        repo_id=llm_model,
        task="text-generation",
        # task="conversational",
        huggingfacehub_api_token=hf_token,
        temperature=0.1, # Low temp = more factual, less creative
        server_kwargs={"wait_for_model": True}, 
    )

    # 6. prompt & Chain (The instructions)
    system_prompt = (
        "Your are an expert overseas master's advisor."
        "Use ONLY the following context to answer the student's question."
        "If the answer is not in the context, say you don't know.\n\n"
        "Context: {context}"
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])

    # 7. Connecting it all
    qa_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(vector_store.as_retriever(), qa_chain)

    # 8. The test query
    user_query = "What is the GPA requirement for NUS?"
    print(f"User: {user_query}")

    response = rag_chain.invoke({"input": user_query})

    # Verification
    print("\n---- AI response ----")
    print(response["answer"])

    print("\n ---- Source documents used ----")
    for i, doc in enumerate(response["context"]):
        print(f"Source{i+1}: {doc.metadata.get('source')}")
        print(f"Snippet: {doc.page_content[:150]}")


if __name__ == "__main__":
    run_mvp()