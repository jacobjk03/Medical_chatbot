from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# function to load pdf file
def load_pdf(data):
    loader = DirectoryLoader(data, 
                    glob="*.pdf",
                    loader_cls=PyPDFLoader)
    
    documents = loader.load()
    return documents


# Creating text chunks from extracted data
def text_spliter(extracted_data):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=20
    )

    text_chunks = text_splitter.split_documents(extracted_data)

    # Add metadata for page + paragraph
    enriched_chunks = []
    for doc in text_chunks:
        page_num = doc.metadata.get("page", "?")
        paragraphs = doc.page_content.split("\n\n")

        for i, para in enumerate(paragraphs):
            if para.strip():
                enriched_chunks.append(
                    Document(
                        page_content=para.strip(),
                        metadata={
                            "source": "Gale Encyclopedia of Medicine (2nd Edition)",
                            "page": page_num,
                            "paragraph": i + 1
                        }
                    )
                )
    return enriched_chunks


#download embedding model
def download_hugging_face_embeddings():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return embeddings
