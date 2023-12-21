from langchain.prompts import PromptTemplate


def get_prompt_template():
    instruction = """
    You are a helpful, respectful legal research assistant for policymakers.
    Always focus on facts and legal aspects.
    Answer given {question} based on CONTEXT:/n/n {context}/n focusing on detailed facts and legal points.
    If you cant frame an answer just say I am not sure
    Question: {question}
    Response: """
        
    prompt_template = PromptTemplate(template=instruction, 
                                     input_variables=["context", "question"],
                                     )
    print('inside get_prompt_template')
    prompt = {"prompt": prompt_template}

    return prompt
    
 