# -*- encoding: utf-8 -*-


from apps.home import blueprint
from flask import render_template, request, jsonify

from jinja2 import TemplateNotFound
from flask import Flask, render_template, render_template_string, request, redirect, url_for
import openai
from Bio import Entrez
import chromadb
from chromadb.utils import embedding_functions
from flask import flash

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
api_key = ["OPENAI_API_KEY"]

app = Flask(__name__)


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


def get_pubmed_papers(query, max_results=10):
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
                
                # Construct the paper's URL
                paper["url"] = base_url + str(paper["pmid"])
                
                # Check for abstract before accessing
                abstracts = record["MedlineCitation"]["Article"].get("Abstract", {}).get("AbstractText", [])
                paper["abstract"] = " ".join([abs_elem for abs_elem in abstracts if abs_elem])
                papers.append(paper)

    except Exception as e:
        error_message = f"There was an issue fetching data from PubMed: {str(e)}"
        return [], error_message

    return papers, None




def research_assistant(medical_history):
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

    for query in search_queries:
        papers, _ = get_pubmed_papers(query, max_results=1)


        if not papers:
            continue

        paper = papers[0]

        # Extract the URL or DOI from the paper object
        paper_url = getattr(paper, 'url', '#')

        # Extract title
        title = paper.get('title', "Title not available")
        print(paper)

        # Summarize the paper using GPTChat
        summary_prompt = f"Provide a brief summary for the paper titled '{title}' which discusses '{paper['abstract'][:100] if 'abstract' in paper else 'Abstract not available.'}...'"
        summary_text = chat_instance.get_gpt3_response(
            user_input=summary_prompt)
        print(summary_prompt)


        summarized_papers.append((title, summary_text, paper_url))

    return search_queries_response, summarized_papers, messages


@blueprint.route('/fetch_results', methods=['POST'])
def fetch_results():
    error_message = None
    papers = []

    try:
        medical_info = request.form['medical_info']
        proposed_treatment = request.form['proposed_treatment']

        print(f"Medical Info Received: {medical_info}")
        print(f"Proposed Treatment Received: {proposed_treatment}")

        # Construct the search query using the provided information
        search_query = construct_search_query(medical_info, proposed_treatment)
        print(f"Constructed Search Query: {search_query}")

        # Use research_assistant to get the summarized papers
        _, summarized_papers, _ = research_assistant(medical_info)

        # If there's an error message
        if not summarized_papers:
            print(f"Error while fetching papers: No summarized papers found.")

        # Render the results section
        return render_template('home/results.html', papers=summarized_papers, error_message=error_message)

    except Exception as e:
        print(f"Error occurred: {str(e)}")
        error_message = "An unexpected error occurred while processing the request."
        return render_template('home/results.html', papers=[], error_message=error_message)
@blueprint.route('/send_message', methods=['POST'])
def send_message():
    user_message = request.form['message']
    
    # Fetch the summarized papers from Chroma DB to use as context
    summarized_papers = check_vector_db(user_message)
    initial_messages = []

    if summarized_papers:
        for paper in summarized_papers:
            if 'abstract' in paper:
                initial_messages.append({'role': 'assistant', 'content': paper['abstract']})

    # Use the GPTChat class to communicate with OpenAI and get a response
    chat_instance = GPTChat(sys_message="You are a helpful medical research assistant.", model=model, initial_messages=initial_messages)
    response = chat_instance.get_gpt3_response(user_message)

    return jsonify({"response": response})





if __name__ == '__main__':
    app.run(debug=True)
