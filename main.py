import openai
from openai.embeddings_utils import get_embedding

from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.chains import RetrievalQA
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.agents import Tool
from langchain.agents import initialize_agent

from sqlalchemy.sql.expression import true
from tqdm.auto import tqdm
from uuid import uuid4
import pinecone

import pandas as pd
import gradio as gr


df = pd.read_csv('resources/MARVL-QnA.csv')

model_name = 'text-embedding-ada-002'
openai_api_key = "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"
embed = OpenAIEmbeddings(
    model = model_name,
    openai_api_key=openai_api_key,
)

pinecone.init(
    api_key='ec647451-aa29-409a-b30d-c5ceaa7c7fa8',
    environment='gcp-starter'
)
index = pinecone.Index('marvlchatbot')
index_name = 'marvlchatbot'

batch_size = 5

texts = []
metadatas = []

for i in tqdm(range(0, len(df), batch_size)):
    i_end = min(len(df), i+batch_size)
    batch = df.iloc[i:i_end]

    metadatas = [{
        'text': str(record['question']),
        'text': str(record['answer'])
    } for j, record in batch.iterrows()]
    documents = batch['answer']
    batch['answer'] = batch['answer'].fillna('')
    documents = batch['answer'].astype(str)
    embeds = embed.embed_documents(documents)
    ids = batch['id'].astype(str)
    index.upsert(vectors=zip(ids, embeds, metadatas))

index = pinecone.GRPCIndex(index_name)
index.describe_index_stats()

text_field = "text"
index = pinecone.Index(index_name)
vectorstore = Pinecone(
    index, embed.embed_query, text_field
)


llm = ChatOpenAI(
    openai_api_key=openai_api_key,
    model_name='gpt-3.5-turbo',
    temperature=0.0
)

conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=5,
    return_messages=True
)

qa = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

tools = [
    Tool(
        name='Knwledge Base',
        func=qa.run,
        description=('Call back on generic chat gpt to gain more context about the query')
    )
]

agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=False,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory
)

with gr.Blocks() as demo:
  chatbot = gr.Chatbot()
  msg = gr.components.Textbox()
  clear = gr.ClearButton([msg, chatbot])

  def respond(message, chat_history):
    bot_message = agent.run(message)
    print(bot_message)
    chat_history.append((message, bot_message))
    return "", chat_history

  msg.submit(respond, [msg, chatbot], [msg, chatbot])

demo.launch(share=True)