from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
import os
from dotenv import load_dotenv


load_dotenv()
PINECONE_API_KEY = os.getenv('PINECONE_API_KEY')

# setting environment variable
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY

print("Import Successfully")

def pinecone_config():
    # configuration pinecone database
    document_store = PineconeDocumentStore(
        index="default",
        namespace="default",
        dimension=768
    )

    return document_store