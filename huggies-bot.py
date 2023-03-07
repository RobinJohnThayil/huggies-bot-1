import streamlit as st
import pandas as pd
import numpy as np
import openai
from PIL import Image

openai.api_key = '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'

embeddings = pd.read_csv("embeddings.csv")
embeddings = embeddings.drop(embeddings.columns[0], axis=1)

def get_embedding(text):
    result = openai.Embedding.create(
      model="text-embedding-ada-002",
      input=text
    )
    return result["data"][0]["embedding"]
def compute_doc_embeddings(df):
    """
    Create an embedding for each row in the dataframe using the OpenAI Embeddings API.
    
    Return a dictionary that maps between each embedding vector and the index of the row that it corresponds to.
    """
    return {
        idx: get_embedding(r.query) for idx, r in df.iterrows()
    }
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
        sim_score.append((contexts.iloc[i][1],vector_similarity(query_embedding,contexts.iloc[i][2:])))
    sim_score.sort(key=lambda x: x[1], reverse=True)
    return sim_score


#models
def davinciC(query):    
    #query = How to feed my baby in the first year
    ss = calc_sim(query, embeddings)
    context = embeddings[embeddings.values == ss[0][0]].iloc[0][0]
    prompt =f"""Answer the question in as many words and as truthfully as possible using the provided context

    Context:
    {context}

    Q: {query}
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
    ss = calc_sim(query, embeddings)
    context = embeddings[embeddings.values == ss[0][0]].iloc[0][0]
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



#UI
st.markdown(
    """
<style>
button[title="View fullscreen"]{
    visibility: hidden;}
.css-1v0mbdj {
    position: fixed;
    top: -15px;
    left: 30px;
    z-index: 200;
}
.css-1avcm0n {
    background: rgb(251 0 0);
    z-index: 100;
}
</style>
""",
    unsafe_allow_html=True,
)
image = Image.open('logo.png')
st.image(image, width=400)

st.sidebar.info('Please choose the model from the dropdown below. The OpenAI API is powered by a diverse set of models with different capabilities. More info here - https://platform.openai.com/docs/models/overview')
st.set_option('deprecation.showfileUploaderEncoding', False)
add_selectbox = st.sidebar.selectbox("Which model would you like to use?", ("gpt-3.5-turbo", "text-davinci-003", "no context - davinci"))
st.title("Newborn & Infants Bot")
st.header('On the day you bring your newborn baby home, life as you know it changes forever. Huggies has put all their tips, techniques and information in one place, to help make newborn baby care as easy as possible for new parents')
if add_selectbox == "gpt-3.5-turbo":
    text1 = st.text_area('Enter your query:')
    output = ""
    if st.button("Ask Huggies Bot"):
        output = turbo(text1)
        st.success(output)
elif add_selectbox == "text-davinci-003":
    text1 = st.text_area('Enter your query:')
    output = ""
    if st.button("Ask Huggies Bot"):
        output = davinciC(text1)
        st.success(output)
elif add_selectbox == "no context - davinci":
    text1 = st.text_area('Enter your query:')
    output = ""
    if st.button("Ask Huggies Bot"):
        output = davinciNC(text1)
        st.success(output)