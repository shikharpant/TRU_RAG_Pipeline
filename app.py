# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import os 
import shutil
import chainlit as cl
from chainlit import user_session
from chainlit import on_message, on_chat_start
from load_embedding import load_embedding
from flask import Flask, render_template, request, jsonify, Response, stream_with_context
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.callbacks.manager import CallbackManager
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from prompt_template import get_prompt_template
from process_llm_response import process_llm_response
from dotenv import load_dotenv
load_dotenv()

if os.path.exists('db'):
    shutil.rmtree('db/')
os.system('unzip db.zip')

# Load embeddings (called only once)
retriever= load_embedding()

@on_chat_start
async def init():
    chain_type_kwargs = get_prompt_template()
    openai_api_base='https://api.endpoints.anyscale.com/v1'
    openai_api_key=os.environ["ANYSCALE_API_TOKEN"]
    local_llm = ChatOpenAI(model = 'mistralai/Mixtral-8x7B-Instruct-v0.1', 
                           openai_api_base=openai_api_base, 
                           openai_api_key=openai_api_key, 
                           temperature=0.4, streaming=True, 
                           max_tokens=15000,
                           verbose=True,
                           callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
                           )
                           
    qa_chain = RetrievalQA.from_chain_type(llm=local_llm,
                                           chain_type="stuff",
                                           retriever=retriever,
                                           return_source_documents=True,
                                           chain_type_kwargs=chain_type_kwargs,
                                           verbose=True,
                                           )
    #store the chain as long as the user session is active
    cl.user_session.set("conversation_chain", qa_chain)

@on_message
async def process_response(res:cl.Message):
        # Read chain from user session variable
        chain = cl.user_session.get("conversation_chain")
        cb = cl.AsyncLangchainCallbackHandler(
             stream_final_answer=True, answer_prefix_tokens=["FINAL", "ANSWER"])
        cb.answer_reached = True
        print("in retrieval QA")
        #res.content to extract the content from chainlit.message.Message
        print(f"res : {res.content}")
        response = await chain.acall(res.content, callbacks=[cb])








