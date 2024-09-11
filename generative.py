# https://haystack.deepset.ai/tutorials/22_pipeline_with_promptnode

from haystack.schema import Document

from haystack.pipelines import Pipeline
from haystack.nodes import BM25Retriever

from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import PromptNode, PromptTemplate, AnswerParser

# Initialize the DocumentStore
document_store = InMemoryDocumentStore(use_bm25=True)

# Sample Persian documents
docs = [
    Document(content="تهران پایتخت ایران است."),
    Document(content="اصفهان یکی از شهرهای تاریخی ایران است."),
    Document(content="زبان رسمی ایران فارسی است."),
    Document(content="ایران دارای تاریخ و فرهنگ غنی است.")
]

# Write documents to the DocumentStore
document_store.write_documents(docs)

# Initialize the Retriever
retriever = BM25Retriever(document_store=document_store)

prompt_node = PromptNode(model_name_or_path="google/flan-t5-base",
                         use_gpu=False,  default_prompt_template='deepset/question-answering')

# Initialize the Generator
pipe = Pipeline()
pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

output = pipe.run(query="پایتخت ایران کجاست؟")

print(output)
