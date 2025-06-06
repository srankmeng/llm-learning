import os
from typing import List
from dotenv import load_dotenv, find_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import BaseOutputParser
from langchain_openai import ChatOpenAI

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


load_dotenv(find_dotenv())

QUERY_PROMPT = """
Instructions: Answer the following question based on the context provided with short answer.
Context : {context}
"""

llm = ChatOpenAI(model="gpt-4o-mini", api_key=os.getenv("OPENAI_API_KEY"))
    
# Read data from the file and split it to generate vectorstore
def read_data_and_convert_to_vector(file_path):
    loader = TextLoader(file_path)
    content = loader.load_and_split()
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(content, embeddings)
    return vectorstore

# Generate response using OpenAI
def generate_response(question, retriever):
    prompt = ChatPromptTemplate.from_messages(
    [
        ("system", QUERY_PROMPT),
        ("human", "{input}"),
    ]
)
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    result = rag_chain.invoke({"input": question})
    return result

if __name__ == "__main__":
    # Path to your text file
    file_path = 'context.txt'
    
    # Read data from the file and convert it to vectorstore
    vectorstore = read_data_and_convert_to_vector(file_path)

    # Save to local file
    # vectorstore.save_local("faiss_index")
    # Load from local file
    # vectorstore = FAISS.load_local(
    #     "faiss_index", 
    #     OpenAIEmbeddings(), 
    #     allow_dangerous_deserialization=True)
    
    # Convert vectorstore to retriever
    if vectorstore is not None:
        retriever = vectorstore.as_retriever()
    else:
        retriever = None
    
    # Generate response
    prompt = "Who won the 2025 Olympics men's running 100m"
    response = generate_response(prompt, retriever)
    print(response)
    print(response["answer"])