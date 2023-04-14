#version: babybot+convo_hist+links
import streamlit as st
import pandas as pd
import numpy as np
import openai
from PIL import Image

image = Image.open('fotor_2023-3-9_15_18_29.png')
st.image(image, width = 180)
st.title("Baby Bot")

st.markdown("""---""")



openai.api_key = st.secrets["OPENAI_API_KEY"]

#loading the dataset to pull context from
embeddings = pd.read_csv("embeddings_wl.csv")
embeddings = embeddings.drop(embeddings.columns[0], axis=1)

def get_embedding(text):
    result = openai.Embedding.create(
      model="text-embedding-ada-002",
      input=text
    )
    return result["data"][0]["embedding"]
def vector_similarity(x, y):
    """
    Returns the similarity between two vectors.
    
    Because OpenAI Embeddings are normalized to length 1, the cosine similarity is the same as the dot product.
    """
    return np.dot(np.array(x), np.array(y))
def calc_sim(query, contexts):
    """
    Find the query embedding for the supplied query, and compare it against all of the pre-calculated document embeddings
    to find the most relevant sections. 
    
    Return the list of document sections, sorted by relevance in descending order.
    """
    query_embedding = get_embedding(query)
    sim_score = []
    for i in range(len(contexts.iloc[0:])):
        sim_score.append((contexts.iloc[i][2],vector_similarity(query_embedding,contexts.iloc[i][3:])))
    sim_score.sort(key=lambda x: x[1], reverse=True)
    return sim_score
def handle_input(
               input_str : str,
    conversation_history : str,
    model : str
                 ):
    """Updates the conversation history and generates a response using one of the models below."""
    # Generate a response using GPT-3
    if(model == 'Customized GPT3'):
        message = davinciC(input_str, conversation_history)
    elif(model == 'Default GPT3'):
        message = davinciNC(input_str,conversation_history)
    elif(model == 'Customized ChatGPT (Experimental)'):
        message = turbo(input_str, conversation_history)

    # Update the conversation history
    phrase = f"Q: {input_str}\nA:{message}\n"
    file = open("convo.txt","a")
    file.write(phrase)
    file.close()
    
    return message
#models
def davinciC(query, conversation_history):    
    #query = How to feed my baby in the first year
    link = ''
    ss = calc_sim(query, embeddings)
    if(st.session_state['count'] == 0):
        st.session_state['context'] = embeddings[embeddings.values == ss[0][0]].iloc[0][0]
        print(st.session_state['context'])
    if ss[0][1] > 0.85:
        link = "and also include the following link in the response:"+ embeddings[embeddings.values == ss[0][0]].iloc[0][1]
    prompt =f"""Answer the question in as many words and as truthfully as possible using the provided context {link}

Context:
{st.session_state['context']}
{conversation_history}
Q:{query}
A:"""
    base_model = "text-davinci-003"
    completion = openai.Completion.create(
        model = base_model,
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        temperature = 0,
    )
    return(completion.choices[0].text)
def davinciNC(query, conversation_history):     
    conversation_history += query
    base_model = "text-davinci-003"
    completion = openai.Completion.create(
        model = base_model,
        prompt = conversation_history,
        max_tokens = 1024,
        n = 1,
        temperature = 0,
    )
    return(completion.choices[0].text)
def turbo(query, conversation_history):
    #query = "How to feed my baby in the first year"
    if(st.session_state['count'] == 0):
        ss = calc_sim(query, embeddings)
        st.session_state['context'] = embeddings[embeddings.values == ss[0][0]].iloc[0][0]
    context = st.session_state['context']+"\n"+conversation_history
    base_model = "gpt-3.5-turbo"
    completion = openai.ChatCompletion.create(
        model = base_model,
        messages=[
            {"role": "system", "content": "You are a chatbot that will provide answers with the help of the assistant."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": context}
        ],
        max_tokens = 1024,
        n = 1,
        temperature = 0,
    )
    return(completion['choices'][0]['message']['content'])
#init conversation history
f = open("convo.txt","a")
f.write("")
f.close()
conversation_history = ''''''
if 'count' not in st.session_state:
    st.session_state['count'] = 0

#UI
st.markdown(
    """
<style>
.css-fblp2m {
    fill: rgb(255 255 255);
}
.css-18ni7ap {
    background: #0f059e;
}
.css-1avcm0n {
    background: #0f059e;
}
</style>
""",
    unsafe_allow_html=True,
)
#image = Image.open('logo.png')
#st.image(image, width=400)

st.sidebar.info('Please choose the model from the dropdown below.')
st.set_option('deprecation.showfileUploaderEncoding', False)
#add_selectbox = st.sidebar.selectbox("Which model would you like to use?", ("gpt-3.5-turbo", "text-davinci-003", "no context - davinci"))
add_selectbox = st.sidebar.selectbox("", ("Customized GPT3", "Default GPT3","Customized ChatGPT (Experimental)"))

 
for count in range(25):
    st.sidebar.markdown("\n")
st.sidebar.markdown("""---""")
  
st.sidebar.caption('Note: Some models have been trained with select public content from www.huggies.com')
#st.sidebar.caption("Please reach out to robin.john@kcc.com for any queries", unsafe_allow_html=False)

st.write('On the day you bring your newborn baby home, life as you know it changes forever. We have put all tips, techniques and information in one place, to help make newborn baby care as easy as possible for new parents')

text1 = st.text_area('Enter your query:')
output = ""
if st.button("Ask The Bot"):
    file = open("convo.txt","r")
    conversation_history = file.read()
    file.close()
    output = handle_input(text1,conversation_history,add_selectbox)
    st.success(output)
    st.session_state['count'] += 1
if st.button("Clear context"):
    st.session_state['count'] = 0
    file = open("convo.txt","w")
    file.write("")
    file.close()
