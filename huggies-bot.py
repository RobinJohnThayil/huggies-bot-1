#version: babybot+convo_hist+links+product_links+matched_content_score+context_aware
from bs4 import BeautifulSoup
import requests
import streamlit as st
import pandas as pd
import numpy as np
import openai
import re
from PIL import Image
import tiktoken
from time import sleep
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
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens
def handle_input(
               input_str : str,
    model : str
                 ):
    """Tries to classify the type of user and generates a response using one the models """
    st.session_state['user_type'] = calculate_context(input_str)
    print(st.session_state['user_type'])
    if(model == 'Customized GPT3'):
        change_context()
        message = davinciC(input_str)
    elif(model == 'Default GPT3'):
        message = davinciNC(input_str)
    elif(model == 'Customized ChatGPT (Experimental)'):
        change_context()
        message = turbo(input_str)

    # Update the conversation history
        
    
    return message
def change_context():
    if(len(st.session_state['con_info']) > 0):
        context_placeholder.info(f"This response is being generated with the help of content taken from huggies.com titled {st.session_state['con_info'][0]}, matched with a score of {round(st.session_state['con_info'][1]*100)}%")
    type_placeholder.info(f"The user is identified as:{st.session_state['user_type']}")
def clear_info():
    if st.session_state['count'] > 0:
        st.session_state['count'] = 0
        del st.session_state['messages']
        try:
            context_placeholder.empty()
            type_placeholder.empty()
            sleep(0.01)
        except:
            pass
        del st.session_state['con_info']
        del st.session_state['hist']
        del st.session_state['user_type']
        del st.session_state['prev_resp']
#models
def davinciC(query):    
    #query = How to feed my baby in the first year
    link = ''
    product = None
    e_token_length = num_tokens_from_string(query, "cl100k_base")
    if(e_token_length > 7000):
        limit = "The prompt has exceeded the token limit set by Openai, please clear the context by pressing the button below"
        return(limit)

    if "Information" in st.session_state['user_type']:
        ss = calc_sim(query, embeddings)
        if(st.session_state['count'] == 0 and ss[0][1]>0.85):
            st.session_state['context'] = embeddings[embeddings.values == ss[0][0]].iloc[0][0]
            st.session_state['con_info'] = [ss[0][0],float(ss[0][1])]
            change_context()
            st.session_state['count'] += 1
            #print(st.session_state['context'])
        if ss[0][1] > 0.85:
            link = "and also include the following link in the response:"+ embeddings[embeddings.values == ss[0][0]].iloc[0][1]
        prompt =f"""Answer the question in as many words and as truthfully as possible using the provided context {link}

Context:
{st.session_state['context']}
{st.session_state['hist']}
Q:{query}
A:"""
        token_length = num_tokens_from_string(prompt, "p50k_base")
        if(token_length > 4000):
            limit = "The prompt has exceeded the token limit set by Openai, please clear the context by pressing the button below"
            return(limit)
        base_model = "text-davinci-003"
        completion = openai.Completion.create(
            model = base_model,
            prompt = prompt,
            max_tokens = 1024,
            n = 1,
            temperature = 0,
        )
        resp = completion.choices[0].text

        phrase = f"Q: {query}\nA:{resp}\n"
        st.session_state['hist'] += phrase
        st.session_state['prev_resp'] = resp

        return(resp)

    elif "Potential" in st.session_state['user_type']:
        product = grab_product(st.session_state['prev_resp']+"\nQ:"+query)
        prompt = f"""
You are a bot that answers customer questions on an ecommerce website.
Q:{query}
A:
Only use the following brand for your response -{product}"""
        message = [{"role": "user", "content": prompt}]
        output = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            messages=message,
        )
        return(output['choices'][0]['message']['content'])
def davinciNC(query):
    base_model = "text-davinci-003"
    completion = openai.Completion.create(
        model = base_model,
        prompt = query,
        max_tokens = 1024,
        n = 1,
        temperature = 0,
    )
    return(completion.choices[0].text)
def turbo(query):
    #query = "How to feed my baby in the first year"
    link = ''
    product = None
    txt = query
    
    st.session_state['messages'].append({"role": "user", "content": query})
    ss = calc_sim(query, embeddings)
    if(st.session_state['count'] == 0 and ss[0][1] > 0.85):
        context = "Please answer with the help of the following context if it is related to the question:\n"
        context += embeddings[embeddings.values == ss[0][0]].iloc[0][0]
        st.session_state['con_info'] = [ss[0][0],float(ss[0][1])]
        change_context()
        st.session_state['messages'].append({"role": "assistant", "content": context})
    if ss[0][1] > 0.85:
        link = "Click this link to get more information:"+ embeddings[embeddings.values == ss[0][0]].iloc[0][1]

    #TOKEN LIMIT
    for i in st.session_state['messages']:
        txt += i["content"]
    prompt_length = num_tokens_from_string(txt, "p50k_base")
    if(prompt_length > 4000):
        limit = "The prompt has exceeded the token limit set by Openai, please clear the context by pressing the button below"
        return(limit)
    
    base_model = "gpt-3.5-turbo"
    try:
        completion = openai.ChatCompletion.create(
            model = base_model,
            messages = st.session_state['messages'],
            max_tokens = 1024,
            n = 1,
            temperature = 0,
        )
    except:
        st.warning("You are being rate limited! OpenAI enforces rate limits on the requests you can make to the API. These are applied over requests-per-minute and tokens-per-minute, please try again in a bit.")
        return("")
    response = completion['choices'][0]['message']['content']
    st.session_state['messages'].append({"role": "assistant", "content": response})
    if ss[0][1] > 0.85:
        product = grab_product(response)
    response += link + "Here's a link to our relavant product:\n" + product
    return(response)

def grab_product(resp):
    #query = "List out potential products from the paragraph below-\n"+resp
    query = "List potential huggies products from the paragraph below,\n"+resp
    output = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a bot that tries to identify products from a paragraph."},
            {"role": "user", "content": query},
            {"role": "assistant", "content": "Only respond with the product"},
            {"role": "assistant", "content": "If there isn't a product respond by saying \"None\""}
        ],
        temperature = 1.5
    )
    model_output = output['choices'][0]['message']['content']
    model_output = re.sub(r'[^\w\s\n]+', '', model_output)
    if "None" in model_output:
        return None
    search = model_output + " huggies amazon"
    print("search term:",search)
    url = 'https://www.google.com/search'

    headers = {
	    'Accept' : '*/*',
	    'Accept-Language': 'en-US,en;q=0.5',
	    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/98.0.4758.82',
    }
    parameters = {'q': search}

    content = requests.get(url, headers = headers, params = parameters).text
    soup = BeautifulSoup(content, 'html.parser')
    search = soup.find(id = 'search')
    links = []
    for link in search.findAll('a'):
        links.append(link.get('href'))
    amazon_links = re.findall(r'https://www\.amazon\.\S*(?=\')', str(links))
    if(len(amazon_links) == 0):
        return None
    else:
        return(amazon_links[0])
def calculate_context(query):
    """Classify a customer as Information-seeking or Potential-buyer or Unsure based on the input query"""
    prompt =f"""
You are a bot that answers customer questions on an ecommerce website.
A customer visting the website will be in one of the below states:
1. Information-seeking: if the customer is just trying to gather information but has no intention to buy the product
2. Potential-buyer: if the customer is interested in buying the product soon.
3. Unsure: if you are unable to estimate the state


Here are some comma-separated examples for two categories:
1. Information-seeking: "I have ... condition. Can you help?", "I need information around ..."
2. Potential-buyer: "What product would you recommend?", "Do you sell ...?"

Your task is to classify a customer as Information-seeking or Potential-buyer or Unsure based on this input:
{query}. Provide only classification as answer
"""
    model="gpt-3.5-turbo"
    messages = [{"role": "user", "content": prompt}]
    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return(response['choices'][0]['message']['content'])


#"""SESSION VARIABLES"""
if 'count' not in st.session_state:
    st.session_state['count'] = 0
if 'context' not in st.session_state:
    st.session_state['context'] = ""
if 'messages' not in st.session_state:
    st.session_state['messages'] = [
        {"role": "system", "content": "You are a helpful chatbot."}
    ]
if 'hist' not in st.session_state:
    st.session_state['hist'] = """"""
if 'con_info' not in st.session_state:
    st.session_state['con_info'] = []
if 'user_type' not in st.session_state:
    st.session_state['user_type'] = ""
if 'prev_resp' not in st.session_state:
    st.session_state['prev_resp'] = ""
#"""END OF SESSION VARIABLES"""

#"""UI"""
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
add_selectbox = st.sidebar.selectbox("", ("Customized GPT3", "Default GPT3","Customized ChatGPT (Experimental)"), on_change=clear_info())
with st.sidebar:
        context_placeholder = st.empty()
        type_placeholder = st.empty()
for count in range(25):
    st.sidebar.markdown("\n")
st.sidebar.markdown("""---""")
  
st.sidebar.caption('Note: Some models have been trained with select public content from www.huggies.com')
#st.sidebar.caption("Please reach out to robin.john@kcc.com for any queries", unsafe_allow_html=False)

	
st.write('On the day you bring your newborn baby home, life as you know it changes forever. We have put all tips, techniques and information in one place, to help make newborn baby care as easy as possible for new parents')

text1 = st.text_area('Enter your query:')
output = ""
if st.button("Ask The Bot"):
    # file = open("convo.txt","r")
    # conversation_history = file.read()
    # file.close()
    with st.spinner(text="In progress..."):    
        output = handle_input(text1,add_selectbox)
        if output == "The prompt has exceeded the token limit set by Openai, please clear the context by pressing the button below":
            st.warning(output)
        else:
            st.success(output)
if st.button("Clear context"):
    clear_info()
