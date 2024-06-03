import streamlit as st
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
import os
import google.generativeai as genai
import os
from dotenv import load_dotenv
import uuid
import base64
import json
import numpy as np

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
db = FAISS.load_local("faiss_index_all", embeddings=embeddings, allow_dangerous_deserialization=True)

def find_match(input):
    input_em = embeddings.embed_query(input)
    result = db.search(input_em, search_type="")
    # return result['matches'][0]['metadata']['text']+"\n"+result['matches'][1]['metadata']['text']
    print(result)

find_match("GL Master Data")

