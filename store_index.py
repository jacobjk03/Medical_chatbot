from src.helper import load_pdf, text_spliter, download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from dotenv import load_dotenv
import os

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')

extracted_data = load_pdf("data")
text_chunks = text_spliter(extracted_data)
embeddings = download_hugging_face_embeddings()

index_name="medical-chatbot"

#Initializing the Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)   

#Creating Embeddings for Each of The Text Chunks & storing
docsearch = PineconeVectorStore.from_documents(text_chunks, embeddings, index_name=index_name)
