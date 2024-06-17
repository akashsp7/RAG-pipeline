import os
os.environ['OPENAI_API_KEY'] = 'YOUR_API_KEY'

import warnings
warnings.filterwarnings('ignore')

from langchain.chains import RetrievalQA
from langchain.vectorstores import Chroma
from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings 

loader = TextLoader('./inception.txt')
document = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)

text = splitter.split_documents(document)

embeddings = OpenAIEmbeddings()
database = Chroma.from_documents(text, embeddings, collection_name='Inception')

llm = OpenAI(temperature=0.1) #Lower temp better here since precise answers are expected
chain = RetrievalQA.from_chain_type(llm, retriever=database.as_retriever())

Questions = ['How does the concept of "shared dreaming" work in the world of "Inception"?',
'What is the significance of the totem, and how does it function for different characters?',
'How does personal guilt of Cobb about Mal affect the dream heists and his interactions with the team?',
'What role does time dilation play in the different levels of dreams in the movie?',
'How does the film explore the theme of reality versus illusion?',
'What are the motivations behind the inception mission assigned to Cobb and his team?',
'How do the different dream levels visually and thematically differ from each other in the movie?',
'What is the significance of the ending scene with the spinning top, and what are some interpretations of it?',
'How do the character backgrounds and skills contribute to the success of the inception mission?',
'How does the soundtrack, composed by Hans Zimmer, enhance the overall atmosphere and storytelling of "Inception"?']

for question in Questions:
    print(f'Question: {question}')
    print(f'Answer: {chain.run(question)}')