import streamlit as st
 
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
from langchain_classic.chains import RetrievalQA
from langchain_community.document_transformers import LongContextReorder
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
 
st.header("Ask me anything")
 
 
@st.cache_resource()
def load_llm():
    llm = LlamaCpp(
        model_path="Client/models/wizardcoder-python-7b-v1.0.Q4_K_S.gguf",
        callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
        verbose=True,
        f16_kv=True,
        n_gpu_layers=-1,
        n_ctx=4096,
        max_tokens=500,
        n_batch=512,
        n_threads=4,
        streaming=True,
        temperature=0.01,
        top_p=0.95,
        top_k=40,
        repeat_penalty=1.2,
        stop=["###"],
    )
    return llm
 
 
@st.cache_resource()
def load_embedding_model():
    return HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={"normalize_embeddings": True},
    )
 
 
@st.cache_resource()
def load_retriever1():
    embedding_model = load_embedding_model()
    vector_store = Chroma(
        persist_directory="Client/vectorDb",
        embedding_function=embedding_model,
    )
    retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={"k": 6})
    return retriever
 
 
@st.cache_resource()
def load_retriever2():
    embedding_model = load_embedding_model()
    vector_store = Chroma(
        persist_directory="Client/vectorDb",
        embedding_function=embedding_model,
    )
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 6})
    return retriever
 
 
@st.cache_resource()
def load_final_retriever():
    retriever1 = load_retriever1()
    retriever2 = load_retriever2()
    ensemble_retriever = EnsembleRetriever(
        retrievers=[retriever1, retriever2], weights=[0.5, 0.5]
    )
    reordering = LongContextReorder()
    pipeline_compressor = DocumentCompressorPipeline(transformers=[reordering])
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=pipeline_compressor, base_retriever=ensemble_retriever
    )
    return compression_retriever
 
 
def load_prompt():
    template = """
### Instruction:
You are an assistant for question-answering tasks. \
Use the following pieces of retrieved context which contain the data for books and Conversations to answer the latest question in Conversations. \
Conversations have the latest question and may also have chat history. \
if you don't know the answer, just say you don't know, DO NOT provide incorrect answer. \
 
### Context:
{context}
 
### Conversations:
{question}
 
### Response:
"""
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    return prompt
 
 
def chains_of_qa():
    llm = load_llm()
    prompt = load_prompt()
    retriever = load_final_retriever()
    chains = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        verbose=True,
        chain_type_kwargs={
            "verbose": True,
            "prompt": prompt,
        },
    )
    return chains
 
 
if "conversation" not in st.session_state:
    st.session_state.conversation = []
 
user_query = st.chat_input("Ask me anything")
chains = chains_of_qa()
 
 
def generate_answer(question):
    if question is None:
        return
 
    sz = len(st.session_state.conversation)
 
    final_query = ""
    if sz > 0:
        for entries in st.session_state.conversation[-2:]:
            final_query = final_query + "{" + "\n"
            final_query = final_query + "Human:" + "\n"
            final_query = final_query + entries["question"] + "\n"
            final_query = final_query + "AI:" + "\n"
            final_query = final_query + entries["answer"] + "\n"
            final_query = final_query + "}" + "\n"
 
    final_query = final_query + "Human:" + "\n"
    final_query = final_query + question
    ai_response = chains.invoke({"query": final_query})["result"]
    st.session_state.conversation.append({
        "question": question,
        "answer": ai_response,
    })
 
 
def display_answer():
    for entries in st.session_state.conversation:
        with st.chat_message("Human"):
            st.write(entries["question"])
        with st.chat_message("AI"):
            st.write(entries["answer"])
 
 
if user_query:
    with st.spinner("Generating ..."):
        generate_answer(user_query)
        display_answer()