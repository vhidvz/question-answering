from haystack.schema import Document

from haystack.nodes import FARMReader, BM25Retriever
from haystack.pipelines import ExtractiveQAPipeline
from haystack.document_stores import InMemoryDocumentStore

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

# Step 3: Initialize the Retriever (BM25)
retriever = BM25Retriever(document_store=document_store)

# xlm-roberta-large
# facebook/m2m100_418M
# google/mt5-base or google/mt5-large

# Step 4: Initialize the Reader using Hugging Face (Multilingual BERT for example)
# Note: We use a multilingual model that supports both Persian and English.
reader = FARMReader(
    model_name_or_path="xlm-roberta-large", model_kwargs={'cache_dir': '.data'}, use_gpu=False)

# Step 5: Create the QA Pipeline
qa_pipeline = ExtractiveQAPipeline(reader=reader, retriever=retriever)

# Step 6: Ask Questions
questions = [
    "What is the capital of Iran?",    # English
    "پایتخت ایران چیست؟"               # Persian
]

for question in questions:
    prediction = qa_pipeline.run(query=question, params={"Retriever": {
                                 "top_k": 5}, "Reader": {"top_k": 3}})
    print(f"Question: {question}")
    for answer in prediction["answers"]:
        print(f"Answer: {answer.answer} (Score: {answer.score})")
