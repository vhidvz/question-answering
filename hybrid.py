# https://haystack.deepset.ai/tutorials/26_hybrid_retrieval

from haystack.nodes import JoinDocuments, SentenceTransformersRanker
from haystack.nodes import EmbeddingRetriever, BM25Retriever
from haystack.schema import Document

from haystack.pipelines import Pipeline
from haystack.nodes import BM25Retriever
from haystack.nodes import PreProcessor
from haystack.document_stores import InMemoryDocumentStore

# Initialize the DocumentStore
document_store = InMemoryDocumentStore(use_bm25=True)

# Sample Persian documents
docs = [
    Document(content="تهران پایتخت ایران است.", meta={"name": "doc1"}),
    Document(content="اصفهان یکی از شهرهای تاریخی ایران است.",
             meta={"name": "doc2"}),
    Document(content="زبان رسمی ایران فارسی است.", meta={"name": "doc3"}),
    Document(content="ایران دارای تاریخ و فرهنگ غنی است.",
             meta={"name": "doc4"})
]

preprocessor = PreProcessor(
    clean_empty_lines=True,
    clean_whitespace=True,
    clean_header_footer=True,
    split_by="word",
    split_length=512,
    split_overlap=32,
    split_respect_sentence_boundary=True,
)

docs_to_index = preprocessor.process(docs)


sparse_retriever = BM25Retriever(document_store=document_store)
dense_retriever = EmbeddingRetriever(
    document_store=document_store,
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    use_gpu=False,
    scale_score=False
)

document_store.delete_documents()
document_store.write_documents(docs_to_index)
document_store.update_embeddings(retriever=dense_retriever)


join_documents = JoinDocuments(join_mode="concatenate")
rerank = SentenceTransformersRanker(
    model_name_or_path="cross-encoder/ms-marco-MiniLM-L-6-v2")


pipeline = Pipeline()
pipeline.add_node(component=sparse_retriever,
                  name="SparseRetriever", inputs=["Query"])
pipeline.add_node(component=dense_retriever,
                  name="DenseRetriever", inputs=["Query"])
pipeline.add_node(component=join_documents, name="JoinDocuments",
                  inputs=["SparseRetriever", "DenseRetriever"])
pipeline.add_node(component=rerank, name="ReRanker", inputs=["JoinDocuments"])

prediction = pipeline.run(
    query="apnea in infants",
    params={
        "SparseRetriever": {"top_k": 10},
        "DenseRetriever": {"top_k": 10},
        "JoinDocuments": {"top_k_join": 15},  # comment for debug
        # "JoinDocuments": {"top_k_join": 15, "debug":True}, #uncomment for debug
        "ReRanker": {"top_k": 5},
    },
)


def pretty_print_results(prediction):
    for doc in prediction["documents"]:
        print(doc.meta["title"], "\t", doc.score)
        print(doc.meta["abstract"])
        print("\n", "\n")


pretty_print_results(prediction)
