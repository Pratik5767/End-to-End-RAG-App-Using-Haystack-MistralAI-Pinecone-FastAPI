from haystack import Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import SentenceTransformersDocumentEmbedder
from haystack_integrations.document_stores.pinecone import PineconeDocumentStore
from haystack.components.converters import PyPDFToDocument
from pathlib import Path
import os
from dotenv import load_dotenv
from QASystem.utility import pinecone_config


def ingest(document_store):
    indexing = Pipeline()

    # adding component in pipeline
    indexing.add_component("converter", PyPDFToDocument())
    indexing.add_component("splitter", DocumentSplitter(split_by="sentence", split_length=2))
    indexing.add_component("embedder", SentenceTransformersDocumentEmbedder())
    indexing.add_component("writer", DocumentWriter(document_store))

    # connecting the components
    indexing.connect("converter", "splitter")
    indexing.connect("splitter", "embedder")
    indexing.connect("embedder", "writer")

    indexing.run({
        "converter": {
            "sources": [Path("D:\\Pratik\\AI\\End to End RAG App Using Haystack MistralAI Pinecone & FastAPI\\data\\Retrieval-Augmented generation for knowledge intensive nlp tasks.pdf")]
        }
    })



if __name__ == "__main__":

    document_store = pinecone_config()
    ingest(document_store)