# these three lines swap the stdlib sqlite3 lib with the pysqlite3 package
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceEmbeddings


def load_corpus():
  loader = CSVLoader(file_path="./innovate_uk_funded_projects_cols.csv")
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=50)
  docs = loader.load_and_split(text_splitter)
  return docs


def embed(docs):
  # Embed and store the texts
  # Supplying a persist_directory will store the embeddings on disk
  persist_directory = 'db'

  model_id = 'sentence-transformers/all-MiniLM-L6-v2'
  model_kwargs = {'device': 'cpu'}
  hf_embedding = HuggingFaceEmbeddings(
      model_name=model_id,
      model_kwargs=model_kwargs
  )
  vectordb = Chroma.from_documents(documents=docs, embedding=hf_embedding, persist_directory=persist_directory)
  return vectordb

print("Loading docs...")
docs = load_corpus()
print("Embedding docs...")
vdb = embed(docs[:10000])
print("Querying...")
res = vdb.similarity_search("machine learning and AI")
print(res)
