# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.schema.messages import HumanMessage, SystemMessage
# from langchain.schema.document import Document
# from langchain.vectorstores import FAISS
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# import os
# import uuid
# import base64
# from fastapi import FastAPI, Request, Form, Response, File, UploadFile
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.encoders import jsonable_encoder
# from fastapi.middleware.cors import CORSMiddleware
# import json
# from dotenv import load_dotenv
# import google.generativeai as genai


# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.schema.messages import HumanMessage, SystemMessage
# from langchain.schema.document import Document
# from langchain.vectorstores import FAISS
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# load_dotenv()

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"],
# )

# api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)
# # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# prompt_template = """
# You are an expert in Financial Accounting SOPs that contains specification, features, etc.
# Answer the question based only on the following context, which can include text, images and tables:
# {context}
# Question: {question}
# Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
# Just return the helpful answer in as much as detailed possible. Do not hallucinate while giving answers.
# Answer:
# """

# qa_chain = LLMChain(llm=ChatGoogleGenerativeAI(model="gemini-pro", max_output_tokens=1024),
#                         prompt=PromptTemplate.from_template(prompt_template))

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/get_answer")
# async def get_answer(question: str = Form(...)):
#     relevant_docs = db.similarity_search(question)
#     context = ""
#     relevant_images = []
#     for d in relevant_docs:
#         if d.metadata['type'] == 'text':
#             context += '[text]' + d.metadata['original_content']
#         elif d.metadata['type'] == 'table':
#             context += '[table]' + d.metadata['original_content']
#         elif d.metadata['type'] == 'image':
#             context += '[image]' + d.page_content
#             relevant_images.append(d.metadata['original_content'])
#     result = qa_chain.run({'context': context, 'question': question})
#     return JSONResponse({"relevant_images": relevant_images[0], "result": result})

# from IPython.display import Markdown
# from langchain.chat_models import ChatOpenAI
# from langchain.embeddings import OpenAIEmbeddings
# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.schema.messages import HumanMessage, SystemMessage
# from langchain.schema.document import Document
# from langchain.vectorstores import FAISS
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# import os
# import uuid
# import base64
# from fastapi import FastAPI, Request, Form, Response, File, UploadFile
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.encoders import jsonable_encoder
# from fastapi.middleware.cors import CORSMiddleware
# import json
# from dotenv import load_dotenv
# import google.generativeai as genai


# from langchain.chains import LLMChain
# from langchain.prompts import PromptTemplate
# from langchain.schema.messages import HumanMessage, SystemMessage
# from langchain.schema.document import Document
# from langchain.vectorstores import FAISS
# from langchain.retrievers.multi_vector import MultiVectorRetriever
# from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
# load_dotenv()

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")

# # Configure CORS
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  
#     allow_credentials=True,
#     allow_methods=["*"], 
#     allow_headers=["*"],
# )

# api_key = os.getenv("GOOGLE_API_KEY")
# genai.configure(api_key=api_key)
# # embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
# embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")

# db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

# prompt_template = """
# You are an expert in Financial Accounting SOPs that contains specification, features, etc.
# Answer the question based only on the following context, which can include text, images and tables:
# {context}
# Question: {question}
# Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
# Just return the helpful answer in as much as detailed possible. Do not hallucinate while giving answers.
# NOTE: Please give output in bullet points.
# Answer:
# """

# qa_chain = LLMChain(llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro", max_output_tokens=1024, convert_system_message_to_human=True),
#                         prompt=PromptTemplate.from_template(prompt_template),
#                     )

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/get_answer")
# async def get_answer(question: str = Form(...)):
#     relevant_docs = db.similarity_search(question)
#     context = ""
#     relevant_images = []
#     for d in relevant_docs:
#         if d.metadata['type'] == 'text':
#             context += '[text]' + d.metadata['original_content']
#         elif d.metadata['type'] == 'table':
#             context += '[table]' + d.metadata['original_content']
#         elif d.metadata['type'] == 'image':
#             context += '[image]' + d.page_content
#             relevant_images.append(d.metadata['original_content'])

#     # Run the question-answering chain
#     result = qa_chain.run({'context': context, 'question': question})

#     # Check if relevant_images is empty before accessing its first element
#     response_data = {
#         "relevant_images": relevant_images[0] if relevant_images else None,
#         "result": result
#     }

#     return JSONResponse(content=response_data)

import streamlit as st
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.schema.document import Document
from langchain.vectorstores import FAISS
from langchain.retrievers.multi_vector import MultiVectorRetriever
import os
import uuid
import base64
import json
from dotenv import load_dotenv
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# Load API key and configure Generative AI
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# Initialize embeddings and FAISS index
embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("faiss_index_all", embeddings, allow_dangerous_deserialization=True)

# Define prompt template for LLMChain
prompt_template = """
You are an expert in Financial Accounting SOPs that contains specification, features, etc.
Answer the question based only on the following context, which can include text, images and tables:
{context}
Question: {question}
Don't answer if you are not sure and decline to answer and say "Sorry, I don't have much information about it."
Just return the helpful answer in as much as detailed possible. 
NOTE: 1.Please give output in bullet points.
      2.Please do NOT include considerations and points to be remembered.
      3. Do not hallucinate while giving answers.

Answer:
"""

# Initialize LLMChain with ChatGoogleGenerativeAI
qa_chain = LLMChain(
    llm=ChatGoogleGenerativeAI(model="gemini-1.5-pro", max_output_tokens=2048, convert_system_message_to_human=True,
                               temperature=0.001,),
    prompt=PromptTemplate.from_template(prompt_template),
)

# Streamlit app
st.title("SOP Assistant")

question = st.text_input("Enter your question:")
if st.button("Get Answer"):
    if question:
        relevant_docs = db.similarity_search(question)
        context = ""
        relevant_images = []
        relevant_tables = []

        for d in relevant_docs:
            if d.metadata['type'] == 'text':
                context += '[text]' + d.metadata['original_content']
            elif d.metadata['type'] == 'table':
                context += '[table]' + d.metadata['original_content']
                relevant_tables.append(d.metadata['original_content'])
            elif d.metadata['type'] == 'image':
                context += '[image]' + d.page_content
                if d.metadata['original_content'] not in relevant_images:
                    relevant_images.append(d.metadata['original_content'])

        # Run the question-answering chain
        result = qa_chain.run({'context': context, 'question': question})

        # Check if relevant_images is empty before accessing its first element
        relevant_image = relevant_images[0] if relevant_images else None

        st.write("Answer:")
        st.markdown(result)

        st.write('Relevant Images: ')
        for image in relevant_images:
            if image.startswith('http'):
                st.image(image)
            else:
                # Assume base64 encoded
                st.image(base64.b64decode(image))
        
        for table in relevant_tables:
            st.write(table)
    else:
        st.error("Please enter a question.")
