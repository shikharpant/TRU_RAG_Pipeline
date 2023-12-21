#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug 16 21:31:41 2023

@author: muski
"""

import textwrap

def wrap_text_preserve_newlines(text, width=110):
    # Split the input text into lines based on newline characters
    lines = text.split('\n')
    # Wrap each line individually
    wrapped_lines = [textwrap.fill(line, width=width) for line in lines]
    # Join the wrapped lines back together using newline characters
    wrapped_text = '\n'.join(wrapped_lines)
    return wrapped_text

def source_metadata(llm_response_sources):
    print('inside source_metadata')
    source_details = []
    for source in llm_response_sources:
        source_ = source.metadata['source'] # Use get method with default value
        source_details.append(source_[19:-4])
    return '\n'.join(source_details)

def process_llm_response(llm_response):
    print('process_llm_response')
    #print(llm_response)
    result_text = '\n' +source_metadata(llm_response["source_documents"]) + '\n'
    print('process_llm_response executed')
    return (result_text)