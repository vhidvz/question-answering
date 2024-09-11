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
    Document(content="تهران پایتخت ایران است.", meta={"name": "doc1"}),
    Document(content="اصفهان یکی از شهرهای تاریخی ایران است.",
             meta={"name": "doc2"}),
    Document(content="زبان رسمی ایران فارسی است.", meta={"name": "doc3"}),
    Document(content="ایران دارای تاریخ و فرهنگ غنی است.",
             meta={"name": "doc4"})
]

# Write documents to the DocumentStore
document_store.write_documents(docs)

# Initialize the Retriever
retriever = BM25Retriever(document_store=document_store)

rag_prompt = PromptTemplate(
    prompt="""Synthesize a comprehensive answer from the following text for the given question.
                             Provide a clear and concise response that summarizes the key points and information presented in the text.
                             Your answer should be in your own words and be no longer than 50 words.
                             \n\n Related text: {join(documents)} \n\n Question: {query} \n\n Answer:""",
    output_parser=AnswerParser(),
)

prompt_node = PromptNode(model_name_or_path="google/mt5-base",
                         default_prompt_template=rag_prompt)

# Initialize the Generator
pipe = Pipeline()
pipe.add_node(component=retriever, name="retriever", inputs=["Query"])
pipe.add_node(component=prompt_node, name="prompt_node", inputs=["retriever"])

output = pipe.run(query="What does Rhodes Statue look like?")

print(output["answers"][0].answer)
