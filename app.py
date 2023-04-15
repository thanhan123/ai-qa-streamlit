import streamlit as st
import pinecone
from langchain import PromptTemplate
import openai
from langchain.llms import OpenAI
from dotenv import load_dotenv
import os

load_dotenv()

# Set the title of the Streamlit app
st.title("Mincolor AI Q&A")

@st.cache_resource
def getPineconeIndex():
  PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
  PINECONE_API_ENV = os.getenv("PINECONE_API_ENV")
  pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
  )
  index_name = "semantic-search-openai-japanese"
  return pinecone.Index(index_name)

@st.cache_resource
def getPromptTemplate():
  template = """Answer the question based on the context below. If the
question cannot be answered using the information provided answer
with "I don't know". Answer in Japanese.

Context: {context}

Question: {query}

Answer: """

  return PromptTemplate(
    input_variables=["query", "context"],
    template=template
  )

@st.cache_resource
def getCustomOpenAI():
  OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
  return OpenAI(
    model_name="gpt-3.5-turbo",
    openai_api_key=OPENAI_API_KEY
  )

embed_model = "text-embedding-ada-002"

query = st.text_input(
  "Enter Your Question: ",
  placeholder = "Ask any medical question",
)

st.cache()
def startQuery(query):
  if len(query) == 0:
    return
  pineconeIndex = getPineconeIndex()
  custom_openai = getCustomOpenAI()
  prompt_template = getPromptTemplate()

  query_embed = openai.Embedding.create(
    input=[query],
    engine=embed_model
  )

  xq = query_embed['data'][0]['embedding']

  res = pineconeIndex.query(xq, top_k=2, include_metadata=True)
  contexts = [item['metadata']['text'] for item in res['matches']]
  context = "\n\n---\n\n".join(contexts)+"\n\n-----\n\n"

  prompt = prompt_template.format(
    context=context,
    query=query
  )
  response = custom_openai(prompt)
  st.text(query)
  st.success(response)

if st.button("enter", type="primary"):
  startQuery(query)

st.markdown('''This AI Q&A has been build by about 40 articles from minacolor.com website.''')
st.markdown('''
For any question outside the knowledge of these articles the web will simply return: I don't know.
Here is the list of articles:''')
st.markdown('''
  'https://minacolor.com/blogs/articles/7751',
  'https://minacolor.com/blogs/articles/7865',
  'https://minacolor.com/blogs/articles/7875',
  'https://minacolor.com/blogs/articles/7890',
  'https://minacolor.com/blogs/articles/7892',
  'https://minacolor.com/blogs/articles/7889',
  'https://minacolor.com/blogs/articles/7887',
  'https://minacolor.com/blogs/articles/7884',
  'https://minacolor.com/blogs/articles/7883',
  'https://minacolor.com/blogs/articles/7882',
  'https://minacolor.com/blogs/articles/7881',
  'https://minacolor.com/blogs/articles/7880',
  'https://minacolor.com/blogs/articles/7878',
  'https://minacolor.com/blogs/articles/7876',
  'https://minacolor.com/blogs/articles/7873',
  'https://minacolor.com/blogs/articles/7872',
  'https://minacolor.com/blogs/articles/7871',
  'https://minacolor.com/blogs/articles/4988',
  'https://minacolor.com/blogs/articles/7746',
  'https://minacolor.com/blogs/articles/7870',
  'https://minacolor.com/blogs/articles/7868',
  'https://minacolor.com/blogs/articles/7867',
  'https://minacolor.com/blogs/articles/7866',
  'https://minacolor.com/blogs/articles/7864',
  'https://minacolor.com/blogs/articles/7838',
  'https://minacolor.com/blogs/articles/7836',
  'https://minacolor.com/blogs/articles/7834',
  'https://minacolor.com/blogs/articles/7829',
  'https://minacolor.com/blogs/articles/7825',
  'https://minacolor.com/blogs/articles/7824',
  'https://minacolor.com/blogs/articles/7823',
  'https://minacolor.com/blogs/articles/7822',
  'https://minacolor.com/blogs/articles/7821',
  'https://minacolor.com/blogs/articles/7820',
  'https://minacolor.com/blogs/articles/7818',
  'https://minacolor.com/blogs/articles/7815',
  'https://minacolor.com/blogs/articles/7812',
  'https://minacolor.com/blogs/articles/7810',
  'https://minacolor.com/blogs/articles/7808',
  'https://minacolor.com/blogs/articles/7806',
  'https://minacolor.com/blogs/articles/7805',
''')
