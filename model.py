import os
import dotenv

from haystack import Document, Pipeline
from haystack.components.writers import DocumentWriter
from haystack.components.joiners import DocumentJoiner
from haystack.components.builders import PromptBuilder
from haystack.components.generators import HuggingFaceLocalGenerator
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors.document_splitter import DocumentSplitter
from haystack_integrations.document_stores.elasticsearch import ElasticsearchDocumentStore
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever, InMemoryEmbeddingRetriever
from haystack.components.embedders import SentenceTransformersTextEmbedder, SentenceTransformersDocumentEmbedder
from haystack_integrations.components.retrievers.elasticsearch import ElasticsearchBM25Retriever, ElasticsearchEmbeddingRetriever


# Load environment variables from .env file
dotenv.load_dotenv()


# Language models
GENERATOR_MODEL = "m3hrdadfi/gpt2-persian-qa"
EMBEDDING_MODEL = 'sentence-transformers/paraphrase-multilingual-mpnet-base-v2'

# Environment variables
ELASTICSEARCH_HOST = os.getenv("ELASTICSEARCH_HOST", None)
ELASTICSEARCH_INDEX = os.getenv("ELASTICSEARCH_INDEX", 'qa_index')

# Prompt templates
PROMPT_TEMPLATE_FA = """
بر مبنای اطلاعات ارائه شده در ادامه به سوال پاسخ بده.

اطلاعات:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

سوال: {{query}}
پاسخ:
"""
PROMPT_SEPARATOR_FA = "پاسخ:"

PROMPT_TEMPLATE_EN = """
Given the context please answer the question.

Context:
{% for document in documents %}
    {{ document.content }}
{% endfor %}

Question: {{query}}
Answer:
"""
PROMPT_SEPARATOR_EN = "Answer:"


class QuestionAnswering():
    def __init__(self):
        self.store = None

        self.writer = None
        self.joiner = None
        self.splitter = None
        self.embedder = {"doc": None, "text": None}

        self.generator = None

        # Retrievers
        self.bm25_retriever = None
        self.embedding_retriever = None

        # Indexing pipeline
        self.indexing_pipeline = None

    def load_model(self):
        self.splitter = DocumentSplitter(
            split_by="word", split_length=512, split_overlap=32)

        self.embedder['doc'] = SentenceTransformersDocumentEmbedder(
            model=EMBEDDING_MODEL)
        self.embedder['doc'].warm_up()
        self.embedder['text'] = SentenceTransformersTextEmbedder(
            model=EMBEDDING_MODEL)
        self.embedder['text'].warm_up()

        if ELASTICSEARCH_HOST:
            self.store = ElasticsearchDocumentStore(
                hosts=ELASTICSEARCH_HOST, index=ELASTICSEARCH_INDEX)

            # Retrievers
            self.bm25_retriever = ElasticsearchBM25Retriever(self.store)
            self.embedding_retriever = ElasticsearchEmbeddingRetriever(
                self.store)

        else:
            self.store = InMemoryDocumentStore()

            # Retrievers
            self.bm25_retriever = InMemoryBM25Retriever(self.store)
            self.embedding_retriever = InMemoryEmbeddingRetriever(self.store)

        self.writer = DocumentWriter(self.store)
        self.joiner = DocumentJoiner(join_mode='merge')

        self.generator = HuggingFaceLocalGenerator(
            model=GENERATOR_MODEL,
            task="text-generation",
            generation_kwargs={"max_new_tokens": 100})
        self.generator.warm_up()

        # Indexing pipelines
        self.indexing_pipeline = Pipeline()
        self.indexing_pipeline.add_component("writer", self.writer)
        self.indexing_pipeline.add_component("splitter", self.splitter)
        self.indexing_pipeline.add_component("embedder", self.embedder['doc'])
        self.indexing_pipeline.connect("embedder", "writer")
        self.indexing_pipeline.connect("splitter", "embedder")

        return self

    def add_documents(self, contents: list[str]):
        documents = [Document(content=content) for content in contents]
        self.indexing_pipeline.run({"documents": documents})

    def answer(self, query: str, lang: str = "fa"):
        prompt = PromptBuilder(template=PROMPT_TEMPLATE_FA) if lang == "fa" else PromptBuilder(
            template=PROMPT_TEMPLATE_EN)
        separator = PROMPT_SEPARATOR_FA if lang == "fa" else PROMPT_SEPARATOR_EN

        basic_rag_pipeline = Pipeline()
        basic_rag_pipeline.add_component("embedder", self.embedder['text'])
        basic_rag_pipeline.add_component(
            "embedding_retriever", self.embedding_retriever)
        basic_rag_pipeline.add_component("bm25_retriever", self.bm25_retriever)
        basic_rag_pipeline.add_component("joiner", self.joiner)
        basic_rag_pipeline.add_component("prompt", prompt)
        basic_rag_pipeline.add_component("llm", self.generator)

        basic_rag_pipeline.connect(
            "embedder", "embedding_retriever.query_embedding")
        basic_rag_pipeline.connect("embedding_retriever", "joiner")
        basic_rag_pipeline.connect("bm25_retriever", "joiner")
        basic_rag_pipeline.connect("joiner", "prompt.documents")
        basic_rag_pipeline.connect("prompt", "llm")

        response = basic_rag_pipeline.run(
            {
                "embedder": {"text": query},
                "embedding_retriever": {"top_k": 3},
                "bm25_retriever": {"query": query, "top_k": 3},
                "joiner": {"top_k": 5},
                "prompt": {"query": query}
            })["llm"]["replies"][0]
        answer = response.split(separator) if response else []
        result = answer.pop() if answer and len(answer) else None
        return result.strip() if isinstance(result, str) else None


if __name__ == "__main__":
    docs = [
        "زبان رسمی ایران فارسی است.",
        "ایران دارای تاریخ و فرهنگ غنی است."
        "اصفهان یکی از شهرهای تاریخی ایران است.",
        "ایران یک کشور چهار فصل و دارای فرهنگی غنی است که پایتخت آن تهران است.",
    ]

    model = QuestionAnswering().load_model()
    model.add_documents(docs)

    print(model.answer("مردم ایران به چه زبانی صحبت می‌کنند؟", "fa"))
