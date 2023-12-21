#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 09:18:25 2023

@author: muski
"""
from langchain.vectorstores import Chroma
from langchain.embeddings import HuggingFaceBgeEmbeddings
from langchain.retrievers import ContextualCompressionRetriever
from load_reranker import BgeRerank 
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain.chat_models import ChatOpenAI
from typing import List
from langchain.chains import LLMChain
from langchain.output_parsers import PydanticOutputParser
from langchain.prompts import PromptTemplate
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv
load_dotenv()


# Output parser will split the LLM result into a list of queries
class LineList(BaseModel):
    # "lines" is the key (attribute name) of the parsed output
    lines: List[str] = Field(description="Lines of text")


class LineListOutputParser(PydanticOutputParser):
    def __init__(self) -> None:
        super().__init__(pydantic_object=LineList)

    def parse(self, text: str) -> LineList:
        lines = text.strip().split("\n")
        return LineList(lines=lines)


output_parser = LineListOutputParser()

QUERY_PROMPT = PromptTemplate(
    input_variables=["question"],
    template="""You are an AI language model assistant. Your task is to generate three 
    different versions of the given user question to retrieve relevant documents from a vector 
    database. By generating multiple perspectives on the user question, your goal is to help
    the user overcome some of the limitations of the distance-based similarity search. 
    Provide these alternative questions separated by newlines.
    Original question: {question}""",
)

# Chain
openai_api_base='https://api.endpoints.anyscale.com/v1'
openai_api_key=os.environ["ANYSCALE_API_TOKEN"]
local_llm = ChatOpenAI(model = 'mistralai/Mistral-7B-Instruct-v0.1', 
                               openai_api_base=openai_api_base, 
                               openai_api_key=openai_api_key, 
                               temperature=0, streaming=True, 
                               max_tokens=2000,
                               verbose=True,)
llm_chain = LLMChain(llm=local_llm, prompt=QUERY_PROMPT, output_parser=output_parser)


def load_embedding():
    model_name = "BAAI/bge-base-en-v1.5"
    reranker_name='BAAI/bge-reranker-large'
    encode_kwargs = {'normalize_embeddings': True} # set True to compute cosine similarity
    model_kwargs={"device": "cpu"}
    bge_embeddings = HuggingFaceBgeEmbeddings(model_name=model_name,
                                                encode_kwargs=encode_kwargs,
                                                model_kwargs=model_kwargs,
                                                )
    persist_directory = 'db'
    embedding = bge_embeddings
    vectordb = Chroma(collection_name='chroma_demo',persist_directory=persist_directory, embedding_function=embedding)
    retriever = vectordb.as_retriever(search_type="mmr", search_kwargs={"k":4})
    multiQ_retriever = MultiQueryRetriever(
        retriever=retriever, llm_chain=llm_chain, parser_key="lines"
        )  # "lines" is the key (attribute name) of the parsed output
    compressor = BgeRerank(top_n=5)
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=multiQ_retriever
        )

    return compression_retriever

    


