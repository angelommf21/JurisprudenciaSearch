#!/usr/bin/env python
# coding: utf-8

# In[3]:


from transformers import AutoTokenizer
import pinecone
import pickle
from collections import Counter
import streamlit as st
import cohere
from sentence_transformers import SentenceTransformer
from datetime import date
from tqdm.auto import tqdm
import pinecone
from collections import Counter
from transformers import AutoTokenizer
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from sentence_transformers import SentenceTransformer
from langchain.prompts import ChatPromptTemplate
from langchain.pydantic_v1 import BaseModel
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from typing import Dict, Any, List


# ### Initialising

# In[4]:


COHERE_API_KEY = 'm4d0XNZxaXRdeAou4DyH1QtrwFvIqbNRoXyJrQln'
co = cohere.Client(COHERE_API_KEY)

pinecone.init(
    api_key="3043c8a5-bd9e-4e4c-897b-82bfb2f3e5bd",
    environment="gcp-starter"
)
idx=pinecone.Index('relacaocoimbra')

with open('descritores_all.pkl', 'rb') as f:
    descritores_all = pickle.load(f)

# ### Sparse Vectors

# In[10]:


class SparseEncoder:
    def __init__(self, model_id):
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

    def build_dict(self, input_batch):
      # store a batch of sparse embeddings
      sparse_emb = []
      # iterate through input batch
      for token_ids in input_batch:
          # convert the input_ids list to a dictionary of key to frequency values
          d = dict(Counter(token_ids))
          # remove special tokens and append sparse vectors to sparse_emb list
          sparse_emb.append({key: d[key] for key in d if key not in [101, 102, 103, 0]})
      # return sparse_emb list
      return sparse_emb

    def generate_sparse_vectors(self, context_batch):
      # create batch of input_ids
      inputs = self.tokenizer(
        context_batch, padding=True,
        truncation=True,
        max_length=512
      )['input_ids']
      # create sparse dictionaries
      sparse_embeds = self.build_dict(inputs)
      return sparse_embeds

    def encode_queries(self, query):
      sparse_vector = self.generate_sparse_vectors([query])[0]
      # Convert the format of the sparse vector
      indices, values = zip(*sparse_vector.items())
      return {"indices": list(indices), "values": list(values)}
    
#model_id = 'neuralmind/bert-base-portuguese-cased'
#sparse_encoder = SparseEncoder(model_id)
#embed = SentenceTransformer('rufimelo/Legal-BERTimbau-sts-large-ma-v3', device='cpu')

@st.cache_data
def load_sparse_encoder(model_id):
    return SparseEncoder(model_id)

@st.cache_resource
def load_sentence_transformer(model_name, device):
    return SentenceTransformer('rufimelo/Legal-BERTimbau-sts-large-ma-v3', device='cpu')

model_id = 'rufimelo/Legal-BERTimbau-sts-large-ma-v3'
sparse_encoder = load_sparse_encoder(model_id)
embed = load_sentence_transformer('rufimelo/Legal-BERTimbau-sts-large-ma-v3', 'cpu')



# --------------------------------------------------------------

# In[12]:


class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __repr__(self):
        return f"Document(page_content='{self.page_content}', metadata={self.metadata})"


# ### Hybrid Search

def hybrid_scale(dense, sparse, alpha: float):
    # check alpha value is in range
    if alpha < 0 or alpha > 1:
        raise ValueError("Alpha must be between 0 and 1")
    # scale sparse and dense vectors to create hybrid search vecs
    hsparse = {
        'indices': sparse['indices'],
        'values':  [v * (1 - alpha) for v in sparse['values']]
    }
    hdense = [v * alpha for v in dense]
    return hdense, hsparse

def hybrid_query(question, top_k, alpha, filter=None):
    def connect_to_pinecone():
        pinecone.init(api_key="3043c8a5-bd9e-4e4c-897b-82bfb2f3e5bd", environment="gcp-starter")
        return pinecone.Index('relacaocoimbra')
    idx = connect_to_pinecone()
    # convert the question into a sparse vector
    sparse_vec = sparse_encoder.generate_sparse_vectors([question])[0]
    sparse_vec = {
        'indices': list(sparse_vec.keys()),
        'values': [float(value) for value in sparse_vec.values()]
    }

    # convert the question into a dense vector
    dense_vec = embed.encode(question).tolist()

    # scale alpha with hybrid_scale
    dense_vec, sparse_vec = hybrid_scale(
    dense_vec, sparse_vec, alpha
    )
    # query pinecone with the query parameters
    result = idx.query(
        vector=dense_vec,
        sparse_vector=sparse_vec,
        top_k=top_k,
        include_metadata=True,
        filter=filter
    )

    # return search results as json
    return result


# ### Reranking


def rerank(hybrid_scale, hybrid_query, query: str, top_k: int, top_n: int, alpha: float, filter=None):
    results = hybrid_query(query, top_k=top_k, alpha=alpha, filter=filter)
    # Filter results
    if results['matches'] == []:
        return 'Não foram encontrados resultados na janela de tempo seleccionada.'
    results_list = [match for match in results['matches']]
    docs_retrieved = []
    for result in results_list:
        result['metadata']['ano'] = int(result['metadata']['ano'])
        result['metadata']['mes'] = int(result['metadata']['mes'])
        result['metadata']['dia'] = int(result['metadata']['dia'])
        if 'context' in result['metadata']:
            doc = Document(result['metadata']['context'], result['metadata'])  
            docs_retrieved.append(doc)

    # Get contexts
    contexts = []
    for doc in docs_retrieved:
        contexts.append(doc.page_content)
    docs = {value: index for index, value in enumerate(contexts, start=0)}
    i2doc = {docs[doc]: doc for doc in docs.keys()}

    # rerank
    rerank_docs = co.rerank(
        query=query, documents=docs, top_n=top_n, model="rerank-multilingual-v2.0"
    )

    reranked_is = []
    for i, doc in enumerate(rerank_docs):
        rerank_i = docs[doc.document["text"]]
        reranked_is.append(rerank_i)
    formated_rerank_docs = []
    
    for i in reranked_is:
        formated_rerank_docs.append(docs_retrieved[i])

    return formated_rerank_docs

def search_and_filter_unique(queries_str: str, min_score: float = 0.85, top_k: int = 10, alpha: int = 1):
    #------------------------------------------------------
    # Perform search using hybrid_query for a string of queries separated by newlines, then filter and retrieve top unique documents based on score.

    # :param queries_str: String containing queries separated by '\n'.
    # :param min_score: Minimum score threshold for filtering results.
    # :param top_k: Number of top results to fetch across all queries.
    # :param alpha: Alpha parameter for hybrid_query.
    # :return: List containing top unique documents based on score across all queries.
    #------------------------------------------------------
    # Split the string into individual queries
    queries = queries_str.strip().split('\n')

    all_results = []
    unique_ids = set()

    for query in queries:
        # Fetch results for each query
        results = hybrid_query(query, top_k=top_k, alpha=alpha)
        all_results.extend(results['matches'])

    # Filter, sort all results based on score, and ensure uniqueness
    filtered_sorted_unique_list = []
    for match in sorted(all_results, key=lambda x: x['score'], reverse=True):
        if match['score'] > min_score and match['id'] not in unique_ids:
            filtered_sorted_unique_list.append(match)
            unique_ids.add(match['id'])

    # Select top documents across all queries
    top_documents = filtered_sorted_unique_list[:top_k]

    # Convert results to Document format
    combined_docs_chain = []
    for result in top_documents:
        page_content = result['metadata']['context']
        metadata = result['metadata']
        metadata.pop('context')
        doc = Document(page_content, metadata)
        combined_docs_chain.append(doc)

    return combined_docs_chain

def remove_initial_characters(text : str) -> str:
    #------------------------------------------------------
    # Function to remove the first 3 characters of each line in a given text.

    # Parameters:
    # text (str): The input text with multiple lines.

    # Returns:
    # str: The text with the first 3 characters of each line removed.
    #------------------------------------------------------
    # Splitting the text into lines
    lines = text.split('\n')

    # Removing the first 3 characters from each line
    modified_lines = [line[3:] if len(line) > 3 else '' for line in lines]

    # Joining the modified lines back into a single text
    modified_text = '\n'.join(modified_lines)

    return modified_text

@st.cache_resource
def connect_to_pinecone():
    pinecone.init(api_key="3043c8a5-bd9e-4e4c-897b-82bfb2f3e5bd", environment="gcp-starter")
    return pinecone.Index('relacaocoimbra')


def run_search_engine_app(hybrid_scale, hybrid_query, rerank):
    idx = connect_to_pinecone()

    # Streamlit app
    st.title('JurisprudênciaSearch - Tribunal da Relação de Coimbra') 

    query = st.text_input("Faça uma pesquisa", key='input')

    options = st.multiselect(
        label='Selecione alguns descritores (opcional):',
        options=descritores_all,
        placeholder='',
        default=[]
    )

    st.markdown("Data do Acórdão:")
    data_i = st.text_input('De (MM/AAAA) (opcional)')
    if data_i != '':
        mes_i = int(data_i[:2])
        ano_i = int(data_i[3:])
        print('Data i:', mes_i, ano_i)

    data_f = st.text_input('a (MM/AAAA) (opcional)')
    if data_f != '':
        mes_f = int(data_f[:2])
        ano_f = int(data_f[3:])
        print('Data f:', mes_f, ano_f)

    if data_i and not data_f:
        today = date.today()
        mes_f = int(today.strftime("%m"))
        ano_f = int(today.strftime("%Y"))

    if st.button('Buscar') and query:
        query = query.lower()
        # Chatbot
        st.subheader('Acórdãos encontrados:')
        if len(options) > 0 and data_i and data_f:
            filter_dict = {
                "$and": [{"descritores": {"$eq": option}} for option in options],
                "$and": [{'ano': {'$gte' : ano_i}}, {'ano' : {'$lte' : ano_f}}] 
            }
            results = rerank(hybrid_scale, hybrid_query, query, top_k=25, top_n=10, alpha=1, filter=filter_dict)
            filtered = []
            for result in results:
                if int(result.metadata['mes']) >= mes_i and int(result.metadata['mes']) <= mes_f:
                    filtered.append(result)
            if filtered == []:
                st.write('Não foram encontrados resultados com o(s) descritore(s) selecionado(s) na janela de tempo seleccionada.')
        elif len(options) > 0 and data_i:
            filter_dict = {
                "$and": [{"descritores": {"$eq": option}} for option in options],
                'ano': {'$gte' : ano_i} 
            }
            results = rerank(hybrid_scale, hybrid_query, query, top_k=25, top_n=10, alpha=1, filter=filter_dict)
            filtered = []
            for result in results:
                if int(result.metadata['mes']) >= mes_i:
                    filtered.append(result)
            if filtered == []:
                st.write('Não foram encontrados resultados com o(s) descritore(s) selecionado(s) na janela de tempo seleccionada.')
        elif len(options) > 0 and data_f:
            filter_dict = {
                "$and": [{"descritores": {"$eq": option}} for option in options],
                'ano' : {'$lte' : ano_f}
            }
            results = rerank(hybrid_scale, hybrid_query, query, top_k=25, top_n=10, alpha=1, filter=filter_dict)
            filtered = []
            for result in results:
                if int(result.metadata['mes']) <= mes_f:
                    filtered.append(result)
            if filtered == []:
                st.write('Não foram encontrados resultados com o(s) descritore(s) selecionado(s) na janela de tempo seleccionada.')
        elif len(options) > 0 and not data_i and not data_f:
            filter_dict = {"$and": [{"descritores": {"$eq": option}} for option in options]}
            results = rerank(hybrid_scale, hybrid_query, query, top_k=25, top_n=10, alpha=1, filter=filter_dict)
            filtered = results
            if filtered == []:
                st.write('Não foram encontrados resultados com o(s) descritore(s) selecionado(s).')
        # No descritores  
        elif len(options) == 0 and data_i and data_f:
            filter_dict = {
                "$and": [{'ano': {'$gte' : ano_i}}, {'ano' : {'$lte' : ano_f}}] 
            }
            results = rerank(hybrid_scale, hybrid_query, query, top_k=25, top_n=10, alpha=1, filter=filter_dict)
            filtered = []
            for result in results:
                if int(result.metadata['mes']) >= mes_i and int(result.metadata['mes']) <= mes_f:
                    filtered.append(result)
            if filtered == []:
                st.write('Não foram encontrados resultados na janela de tempo seleccionada.')
        elif len(options) == 0 and data_i:
            filter_dict = {
                'ano': {'$gte' : ano_i} 
            }
            results = rerank(hybrid_scale, hybrid_query, query, top_k=25, top_n=10, alpha=1, filter=filter_dict)
            filtered = []
            for result in results:
                if int(result.metadata['mes']) >= mes_i:
                    filtered.append(result)
            if filtered == []:
                st.write('Não foram encontrados resultados na janela de tempo seleccionada.')
        elif len(options) == 0 and data_f:
            filter_dict = {
                'ano' : {'$lte' : ano_f}
            }
            results = rerank(hybrid_scale, hybrid_query, query, top_k=25, top_n=10, alpha=1, filter=filter_dict)
            filtered = []
            for result in results:
                if int(result.metadata['mes']) <= mes_f:
                    filtered.append(result)
            if filtered == []:
                st.write('Não foram encontrados resultados na janela de tempo seleccionada.')
        else:
            results = rerank(hybrid_scale, hybrid_query, query, top_k=25, top_n=10, alpha=1)
            filtered = results

        k = 0
        sumarios = []
        # Display reference expanders
        for doc in filtered:
            sumario = doc.metadata['sumario']
            descritores = doc.metadata['descritores']
            ano = str(int(doc.metadata['ano']))
            mes = str(int(doc.metadata['mes']))
            if len(mes) == 1:
                mes = '0' + mes
            dia = str(int(doc.metadata['dia']))
            if len(dia) == 1:
                dia = '0' + dia
            site = doc.metadata['site']
            if sumario not in sumarios:
                sumarios.append(sumario)
                k += 1
                with st.expander(f"Referência {k}"):
                    st.markdown(f"Data do Acórdão: {dia}/{mes}/{ano} (DD/MM/AAAA)")
                    st.markdown(f"Descritores: {descritores}")
                    st.markdown(f"Sumário): {sumario}")
                    st.markdown(f"Link: {site}")

run_search_engine_app(hybrid_scale, hybrid_query, rerank)


# 
