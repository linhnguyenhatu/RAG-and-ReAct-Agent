

from langchain.document_loaders.pdf import PyPDFDirectoryLoader as PDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.vectorstores.chroma import Chroma
from langchain_community.embeddings.ollama import OllamaEmbeddings
from langchain.prompts import ChatPromptTemplate
#from langchain.llms.ollama import Ollama

Path = "Data"

list_of_chunks = None

def load_pdfs():
    loader = PDFLoader(Path)
    return loader.load()
list_of_docs = load_pdfs()

def split_docs():
    spliter = RecursiveCharacterTextSplitter(
        chunk_size = 800,
        chunk_overlap = 80,
        length_function = len,
        is_separator_regex= False)
    return spliter.split_documents(list_of_docs)
list_of_chunks = split_docs()

def create_id(current_chunk, count):
    return current_chunk.metadata['source'] + str(current_chunk.metadata['page']) + str(count)

def create_unique_ids(list_of_chunks):
    count = 0
    current_page = 1
    for i in range(len(list_of_chunks)):
        current_chunk = list_of_chunks[i]
        if int(current_chunk.metadata["page_label"]) > int(current_page):
            count = 0
        current_chunk_id = create_id(current_chunk, count)
        current_chunk.metadata["id"] = current_chunk_id
        count += 1
        current_page = current_chunk.metadata["page_label"]
    return list_of_chunks

list_of_chunks = create_unique_ids(list_of_chunks)

def store_data(list_of_chunks):
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
    dataloader = Chroma(
        persist_directory="CHROMA_PATH", embedding_function=embeddings
    )
    dataloader.persist()
    ids = [list_of_chunks[i].metadata["id"] for i in range(len(list_of_chunks))]
    dataloader.add_documents(list_of_chunks, ids = ids)
    return dataloader

dataloader = store_data(list_of_chunks)

format_prompt = """
Answer the question based only on the following context:

{context}

---

Answer the question based on the above context: {question}
"""


def search_in_docs(question):

    results = dataloader.similarity_search_with_score(question, k = 5)
    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(format_prompt)
    return context_text


#prompt = prompt_template.format(context=context_text, question=question)
#model = Ollama(model = "mistral")
#answer = model.invoke(prompt)
#print(answer)


