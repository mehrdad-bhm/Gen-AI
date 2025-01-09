# Retrieval-Augmented Generation (RAG) with ChromaDB and LLMs

## Introduction

This repository demonstrates a comprehensive implementation of a **Retrieval-Augmented Generation (RAG)** pipeline. RAG combines similarity search techniques with large language models (LLMs) to generate contextually accurate and informed responses.

A key feature of RAG is **similarity search**, which retrieves the most relevant context from a collection of documents based on vector embeddings. The retrieved context is then used as input for an LLM to produce accurate and tailored responses. In this implementation, **ChromaDB** is used as the vector database for efficient similarity search, with options to use **FAISS** for high-performance alternatives.

Additionally, this code highlights three approaches to using LLMs:
1. **Hugging Face Endpoint**
2. **Hugging Face Pipeline**
3. **OpenAI API**

These approaches provide flexibility depending on the scale and requirements of your project.

---

## Purpose of the Code

The purpose of this code is to:

1. **Ingest Documents**: Load and preprocess a collection of documents (e.g., PDFs).
2. **Chunk Documents**: Split large documents into smaller, manageable pieces.
3. **Create a Vector Database**: Store document chunks as vector embeddings in ChromaDB.
4. **Perform Similarity Search**: Retrieve the most relevant context for a given query using vector-based similarity search.
5. **Integrate with LLMs**: Use an LLM to generate contextually accurate responses based on the retrieved context.

---

## Features

- **Document Preprocessing**: Efficiently load and split documents into smaller chunks for processing.
- **Vector Database**: Use ChromaDB to store and retrieve embeddings for similarity search. Optionally, use FAISS for high-performance needs.
- **Similarity Search**: Find the most relevant context for a query using vector similarity.
- **LLM Integration**: Generate responses using LLMs via Hugging Face Endpoint, Hugging Face Pipeline, or OpenAI API.

---

## How Similarity Search Works

1. **Embedding the Documents**: Each document is converted into a vector representation using an embedding model.
2. **Storing the Embeddings**: The vector representations are stored in a vector database (e.g., ChromaDB or FAISS).
3. **Query Embedding**: When a query is submitted, it is also converted into a vector.
4. **Finding Similar Vectors**: The query vector is compared with the document vectors in the database to find the most similar entries.
5. **Retrieving Context**: The most relevant document chunks are retrieved and provided as context for the LLM.

---

## Steps in the Code

### 1. Setting Up the Environment
Install the required libraries:

```bash
!pip install langchain langchain_community pypdf chromadb langchain_huggingface openai tiktoken huggingface_hub accelerate
```

### 2. Loading Documents

Load documents (e.g., PDFs) from the specified directory:

```python
def load_documents():
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    return document_loader.load()
```

### 3. Chunking Documents

Split large documents into smaller chunks to optimize processing:

```python
def split_text(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=400,
        chunk_overlap=100,
        length_function=len,
        add_start_index=True,
    )
    chunks = text_splitter.split_documents(documents)
    return chunks
```

### 4. Creating a Vector Database

Store document chunks as vector embeddings in ChromaDB:

```python
def save_to_chroma(chunks):
    db = Chroma.from_documents(
        chunks, HuggingFaceEmbeddings(), persist_directory=CHROMA_PATH
    )
    db.persist()
```

### 5. Performing Similarity Search

Query the database for relevant document chunks:

```python
db = Chroma(persist_directory=CHROMA_PATH, embedding_function=HuggingFaceEmbeddings())
results = db.similarity_search_with_relevance_scores(query_text, k=3)
```

### 6. Choosing an LLM

This implementation supports three ways to use LLMs:

#### **Option 1: Hugging Face Endpoint**
```python
from langchain_huggingface import HuggingFaceEndpoint
llm = HuggingFaceEndpoint(repo_id="HuggingFaceH4/zephyr-7b-beta")
```

#### **Option 2: Hugging Face Pipeline**
```python
from langchain_huggingface import HuggingFacePipeline
llm = HuggingFacePipeline.from_model_id(
    model_id="HuggingFaceH4/zephyr-7b-beta",
    task="text-generation",
    pipeline_kwargs={"max_new_tokens": 512},
)
```

#### **Option 3: OpenAI API**
```python
from langchain.chat_models import ChatOpenAI
llm = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=os.environ["OPENAI_API_KEY"],
    max_tokens=512,
)
```

### 7. Generating a Response

Format the retrieved context and query into a prompt and generate a response:

```python
prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
prompt = prompt_template.format(context=context_text, question=query_text)
response_text = model.predict(prompt)
```

---

## Differences Between LLM Approaches

- **Hugging Face Endpoint**:
  - Cloud-based.
  - Ideal for production-grade systems.
- **Hugging Face Pipeline**:
  - Local execution.
  - Best for prototyping and experimentation.
- **OpenAI API**:
  - Cloud-based.
  - Provides access to state-of-the-art models like GPT-4.

---

## Input and Output Examples

### Input Example
```plaintext
Query: "Explain how to discard structure results."
```

### Output Example
```plaintext
Response: "To discard structure results, ensure that the data processing pipeline removes unnecessary components based on the context provided."

Sources: ["document1.pdf", "document2.pdf"]
```
