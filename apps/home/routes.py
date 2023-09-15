# -*- encoding: utf-8 -*-


import os
from apps.home import blueprint
from flask import render_template, request, jsonify
from jinja2 import TemplateNotFound
from flask import Flask, render_template, render_template_string, request, redirect, url_for
import openai
from Bio import Entrez
import chromadb
from chromadb.utils import embedding_functions
from flask import flash
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from apps import socketio
from dotenv import load_dotenv

load_dotenv()
api_key = os.environ["OPENAI_API_KEY"]
openai.api_key = api_key


@blueprint.route('/')
@blueprint.route('/index')
def index():

    return render_template('home/index.html', segment='index')


@blueprint.route('/<template>')
def route_template(template):

    try:

        if not template.endswith('.html'):
            template += '.html'

        # Detect the current page
        segment = get_segment(request)

        # Serve the file (if exists) from app/templates/home/FILE.html
        return render_template("home/" + template, segment=segment)

    except TemplateNotFound:
        return render_template('home/page-404.html'), 404

    except:
        return render_template('home/page-500.html'), 500


# Helper - Extract current page name from request
def get_segment(request):

    try:

        segment = request.path.split('/')[-1]

        if segment == '':
            segment = 'index'

        return segment

    except:
        return None


model = "gpt-3.5-turbo-16k"
CHROMA_HOST = "20.241.214.59"
CHROMA_PORT = "8000"


# Initialize ChromaDB client
chroma_client = chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
collection = chroma_client.get_or_create_collection("medical_research_papers")
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2")

# Set up the email and tool name for the Entrez system
Entrez.email = "clay_campbell@hakkoda.io"
Entrez.tool = "MedResearchAid"


class GPTChat:
    def __init__(self, sys_message, model='gpt-3.5-turbo-16k', initial_messages=None):
        self.messages = [{'role': 'system', 'content': sys_message}]
        if initial_messages:
            self.messages.extend(initial_messages)
        self.model = model

    def add_message(self, role, content):
        self.messages.append({'role': role, 'content': content})

    def get_gpt3_response(self, user_input):
        self.add_message('user', user_input)

        response = openai.ChatCompletion.create(
            model=self.model,
            messages=self.messages,
            temperature=0.3,
            stream=True
        )

        responses = ""

        for chunk in response:
            response_content = chunk.get("choices", [{}])[
                0].get("delta", {}).get("content")
            if response_content:
                responses += response_content
                print(response_content, end='', flush=True)

        self.add_message('assistant', responses)

        return responses


def check_vector_db(query):
    """Check if the query results are already in the vector DB."""
    embeddings = sentence_transformer_ef([query])
    results = collection.query(query_embeddings=embeddings[0], n_results=5)
    if results and 'documents' in results and len(results['documents']) > 0:
        return results['documents'][0]
    return None


def construct_search_query(medical_info, proposed_treatment):
    """Use GPT to construct a refined search query based on user input."""
    messages = [
        {"role": "system", "content": "You are a helpful medical research assistant."},
        {"role": "user", "content": f"Given the medical information: '{medical_info}' and the proposed treatment: '{proposed_treatment}', what would be an appropriate PubMed search query?"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo-16k",
        messages=messages,
        max_tokens=100,
        stop=None,
        temperature=0.7
    )
    return response.choices[0].message['content'].strip()


def store_papers_to_db(papers, query):
    """Store papers and their embeddings to the vector DB."""
    papers_embeddings = sentence_transformer_ef(papers)
    collection.upsert(embeddings=papers_embeddings,
                      documents=papers, ids=[query for _ in papers])


def get_pubmed_papers(query, max_results=10, char_limit=500):
    try:
        # Use Entrez.esearch to get PubMed IDs for the given query
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results)
        result = Entrez.read(handle)
        handle.close()
        id_list = result["IdList"]

        # Fetch details for each paper using Entrez.efetch
        papers = []
        base_url = "https://pubmed.ncbi.nlm.nih.gov/"
        if id_list:
            handle = Entrez.efetch(db="pubmed", id=id_list, retmode="xml")
            records = Entrez.read(handle)["PubmedArticle"]
            handle.close()

            for record in records:
                paper = {}
                paper["pmid"] = record["MedlineCitation"]["PMID"]
                paper["title"] = record["MedlineCitation"]["Article"]["ArticleTitle"]
                paper["url"] = base_url + str(paper["pmid"])

                # Check for abstract before accessing
                abstracts = record["MedlineCitation"]["Article"].get(
                    "Abstract", {}).get("AbstractText", [])
                paper["abstract"] = " ".join(
                    [abs_elem for abs_elem in abstracts if abs_elem])

                # Check for link-outs (this is a basic example and might need refining)
                linkouts = record["PubmedData"].get("LinkOutList", [])
                paper["linkouts"] = [link["URL"] for link in linkouts if "URL" in link]

                # Decide whether to summarize
                if len(paper["abstract"]) > char_limit:
                    paper["summary_required"] = True
                else:
                    paper["summary_required"] = False

                papers.append(paper)
                
                # Print the paper details
                print("Paper Title:", paper["title"])
                print("Abstract:", paper["abstract"])
                print("URL:", paper["url"])
                print("Linkouts:", paper["linkouts"])
                print("Summary Required:", paper["summary_required"])
                print("-----------------------------")

    except Exception as e:
        error_message = f"There was an issue fetching data from PubMed: {str(e)}"
        return [], error_message

    return papers, None


def research_assistant_generator(medical_history, chunk_size=2):
    """Assist in finding and summarizing medical research papers based on a detailed medical history."""

    # Initialize summarized_papers and messages lists
    summarized_papers = []
    messages = []

    # Step 1: Analysis Phase - Generate Search Queries
    analysis_prompt = f"Given the following detailed medical history, generate relevant search queries:\n\n{medical_history}"

    chat_instance = GPTChat(
        sys_message="Starting medical paper search...", model=model)
    search_queries = chat_instance.get_gpt3_response(
        user_input=analysis_prompt)
    search_queries = search_queries.split('\n')

    # Store the generated search queries message
    search_queries_response = "Based on the provided information, I will be searching for the following queries:\n" + \
        "\n".join(search_queries)
    messages.append(search_queries_response)

    # Process search queries in chunks
    for i in range(0, len(search_queries), chunk_size):
        chunk_queries = search_queries[i:i+chunk_size]

        for query in chunk_queries:
            papers, _ = get_pubmed_papers(query, max_results=1)

            if not papers:
                continue

            for paper in papers:
                title = paper.get('title', "Title not available")

                # Decide whether to summarize
                if paper["summary_required"]:
                    # Summarize the paper using GPTChat
                    summary_prompt = f"Provide a brief summary for the paper titled '{title}' which discusses '{paper['abstract'][:100]}...'"
                    summary_text = chat_instance.get_gpt3_response(user_input=summary_prompt)
                else:
                    summary_text = paper["abstract"]

                yield title, summary_text, paper["url"], paper.get("linkouts", [])







@socketio.on('fetch_results')
def handle_fetch_results(data):
    print("Handling fetch_results Socket.IO event.")
    medical_info = data['medical_info']
    proposed_treatment = data['proposed_treatment']

    for paper_title, paper_summary, paper_url, paper_linkouts in research_assistant_generator(medical_info):
        emit('new_summary', {
            'paper_title': paper_title, 
            'paper_summary': paper_summary,
            'paper_url': paper_url,
            'paper_linkouts': paper_linkouts
        })
        socketio.sleep(0)  # Yield control to allow other events to be processed

    # Indicate the end of streaming
    emit('streaming_complete')



@socketio.on('user_message')
def handle_user_message(data):
    user_message = data['message']
    # You can use your existing logic to fetch summarized papers, create the context, etc.
    # Fetch the summarized papers from Chroma DB to use as context
    summarized_papers = check_vector_db(user_message)
    initial_messages = []

    if summarized_papers:
        for paper in summarized_papers:
            if 'abstract' in paper:
                initial_messages.append(
                    {'role': 'assistant', 'content': paper['abstract']})

    chat_instance = GPTChat(sys_message="You are a helpful medical research assistant.",
                            model=model, initial_messages=initial_messages)
    response = chat_instance.get_gpt3_response(user_message)

    # Emit the AI's response to the client
    emit('ai_response', {'response': response})



if __name__ == '__main__':
    socketio.run(app)
