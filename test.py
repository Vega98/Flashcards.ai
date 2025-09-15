import tempfile
import os
from pathlib import Path
import random
import streamlit as st

from langchain_ollama import OllamaLLM, OllamaEmbeddings

from langchain.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain.vectorstores import Chroma

from langchain_core.prompts import PromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain.chains import create_retrieval_chain

st.title("Flashcards.ai - RAG Demo")

# Models / hyperparameters
LLM_MODEL = "gemma3"
EMBEDDING_MODEL = "embeddinggemma"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 200

llm = OllamaLLM(model=LLM_MODEL)
embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)

# Upload and index document
uploaded_file = st.file_uploader("Upload a document (PDF or TXT)", type=["pdf", "txt"])
persist_dir = "chroma_db"

if uploaded_file:
    st.info("Processing document...")
    # Persist directory so we don't re-index on every query
    if not Path(persist_dir).exists():
        # Save upload to temp file 
        suffix = ".pdf" if uploaded_file.type == "application/pdf" else ".txt"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # Load document
        if suffix == ".pdf":
            loader = PyPDFLoader(tmp_path)
        else:
            loader = TextLoader(tmp_path)

        document = loader.load()
        splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        chunks = splitter.split_documents(document)

        # Build and persist vector store
        vectordb = Chroma.from_documents(chunks, embeddings, persist_directory=persist_dir)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5}) # retrieve top 5 hits (top-k approach)

        st.success(f"Indexed {len(chunks)} chunks into Chroma (persist dir: {persist_dir}).")
        os.unlink(tmp_path)  # Clean up temp file
    else:
        # Load existing persistent store
        vectordb = Chroma(persist_directory=persist_dir, embedding_function=embeddings)
        retriever = vectordb.as_retriever(search_kwargs={"k": 5}) # retrieve top 5 hits (top-k approach)
        st.info("Loaded existing Chroma index from disk")

    # Build the combine-docs chain + retrieval chain
    # prompt must expect a 'context' variable (create_stuff_documents_chain will pass retrieved docs into 'context')

    prompt = PromptTemplate.from_template(
        "Use ONLY the following context to answer the question. If the answer is not in the context, say you don't know.\n\n"
        "Context:\n{context}\n\nQuestion: {input}\nAnswer:"
    )

    # create_stuff_documents_chain acccepts the llm and a prompt (it formats the retrieved docs into 'context')
    combine_docs_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)

    # create_retrieval_chain stitches retriever -> combine_docs_chain
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    # Query UI
    query = st.text_input("Enter your query (or type 'flashcards' to generate flashcards)")

    if st.button("Ask LLM"):
        if not query:
            st.warning("Please enter a query.")
        else:
            # Flashcard flow: retrieve -> ask LLM to generate flashcards
            if "flashcards" in query.lower():
                all_docs = vectordb.get(include=["documents"])["documents"]
                # Sample up to 5 random docs
                num_samples = min(len(all_docs), 5)
                docs = random.sample(all_docs, num_samples)
                context = "\n\n".join(docs)
                flash_prompt = (
                    "From the context below, generate 5 concise flashcards (Q: ... / A: ...)" \
                    "with short answers (1-2 sentences each). Keep them simple and focused.\n\n"
                    f"{context}" 
                )
                with st.spinner("Generating flashcards..."):
                    response = llm.invoke(flash_prompt)
                st.subheader("Flashcards")
                st.markdown(response)

            else:
                # Standard rag flow using create_retrieval_chain
                with st.spinner("Retrieving and generating answer..."):
                    # .invoke returns a dict containing at least 'answer' and 'context'
                    out = retrieval_chain.invoke({"input": query})
                    answer = out.get("answer") or out.get("output") or str(out) 
                st.subheader("Answer (RAG)")
                st.write(answer)

    # Clean up temp file
    try: 
        os.unlink(tmp_path)
    except Exception:
        pass