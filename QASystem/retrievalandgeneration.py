from haystack.utils import Secret
from haystack.components.embedders import SentenceTransformersTextEmbedder
from haystack.components.builders import PromptBuilder
from haystack_integrations.components.retrievers.pinecone import PineconeEmbeddingRetriever
from haystack.components.generators import HuggingFaceAPIGenerator
from haystack import Pipeline
from QASystem.ingestion import ingest
from QASystem.utility import pinecone_config
import os
from dotenv import load_dotenv

load_dotenv()


prompt_template = """Answer the following query based on the provided context. If the context does
                     not include an answer, reply with 'I don't know'.\n
                     Query: {{query}}
                     Documents:
                     {% for doc in documents %}
                        {{ doc.content }}
                     {% endfor %}
                     Answer: 
                  """


def get_result(query):
    query_pipeline = Pipeline()

    # adding component in pipeline
    query_pipeline.add_component("text_embedder", SentenceTransformersTextEmbedder())
    query_pipeline.add_component("retriever", PineconeEmbeddingRetriever(document_store=pinecone_config()))
    query_pipeline.add_component("prompt_builder", PromptBuilder(template=prompt_template))
    query_pipeline.add_component("llm", HuggingFaceAPIGenerator(api_params={"model": "mistralai/Mistral-7B-v0.1"}, token=Secret.from_token(os.getenv("HF_TOKEN"))))

    # connecting the components
    query_pipeline.connect("text_embedder.embedding", "retriever.query_embedding")
    query_pipeline.connect("retriever.documents", "prompt_builder.documents")
    query_pipeline.connect("prompt_builder", "llm")

    query = query

    results = query_pipeline.run(
        {
            "text_embedder": { "text": query },
            "prompt_builder": { "query": query }
        }
    )

    return results['llm']['replies'][0]



if __name__=="__main__":
    result = get_result("What is a rag token?")
    print(result)