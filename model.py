from haystack.schema import Document

from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import FARMReader, TransformersReader, BM25Retriever, EmbeddingRetriever

# Step 1: Set up the Document Store
document_store = InMemoryDocumentStore(use_bm25=True)

# Step 2: Add some documents (in Persian and English)
docs = [
    Document(content="Hello, this is a document written in English."),
    Document(content="سلام، این یک سند است که به زبان فارسی نوشته شده است."),
    Document(content="The capital of Iran is Tehran."),
    Document(content="پایتخت ایران تهران است.")
]

# Write documents to the Document Store
document_store.write_documents(docs)

# sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
# Step 3: Initialize the EmbeddingRetriever
# retriever = EmbeddingRetriever(
#     'sentence-transformers/paraphrase-multilingual-mpnet-base-v2', document_store=document_store)
# document_store.update_embeddings(retriever)

retriever = BM25Retriever(document_store=document_store)

# Step 4: Initialize the Reader using Hugging Face (Multilingual BERT for example)
# Note: We use a multilingual model that supports both Persian and English.
# reader = FARMReader(top_k_per_candidate=5, top_k_per_sample=5,
#                     model_name_or_path="xlm-roberta-large", model_kwargs={'cache_dir': '.data'}, use_gpu=False)
reader = TransformersReader(
    model_name_or_path="xlm-roberta-large", tokenizer="xlm-roberta-large", use_gpu=False)

# Step 5: Create the QA Pipeline
qa_pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# Step 6: Ask Questions
questions = [
    "What is the capital of Iran?",    # English
    "پایتخت ایران چیست؟"               # Persian
]

for question in questions:
    final = None

    prediction = qa_pipeline.run(
        query=question,
        params={
            # Adjusted top_k to match the number of documents
            "Retriever": {"top_k": 1},
            # Reduced top_k for Reader to improve precision
            "Reader": {"top_k": 5}
        })
    print(f"Question: {question}")

    for answer in prediction["answers"]:
        if final is None or answer.score > final.score:
            final = answer
        print(f"Answer: {answer.answer} (Score: {answer.score})")

    if final is not None:
        print(f"Final Answer: {final.answer}")
