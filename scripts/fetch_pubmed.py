import os
from dotenv import load_dotenv
import requests
import time
import json

# List of topics to search for
TOPICS = [
    "diabetes",
    "hypertension",
    "asthma",
    "breast cancer",
    "COVID-19"
]

N_ABSTRACTS_PER_TOPIC = 100
EMAIL = os.getenv("PUBMED_EMAIL")  # Replace with your email (NCBI requires this for API usage)

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'pubmed', 'processed')
os.makedirs(OUTPUT_DIR, exist_ok=True)

SEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
FETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

all_results = {}

for topic in TOPICS:
    print(f"Fetching PubMed abstracts for topic: {topic}")
    # Step 1: Search for PMIDs
    params = {
        "db": "pubmed",
        "term": topic,
        "retmax": N_ABSTRACTS_PER_TOPIC,
        "retmode": "json",
        "email": EMAIL
    }
    resp = requests.get(SEARCH_URL, params=params)
    resp.raise_for_status()
    idlist = resp.json()["esearchresult"]["idlist"]
    print(f"Found {len(idlist)} articles for {topic}")
    # Step 2: Fetch details for each PMID
    abstracts = []
    if idlist:
        fetch_params = {
            "db": "pubmed",
            "id": ",".join(idlist),
            "retmode": "xml",
            "email": EMAIL
        }
        fetch_resp = requests.get(FETCH_URL, params=fetch_params)
        fetch_resp.raise_for_status()
        from xml.etree import ElementTree as ET
        root = ET.fromstring(fetch_resp.content)
        for article in root.findall(".//PubmedArticle"):
            try:
                pmid = article.findtext(".//PMID")
                title = article.findtext(".//ArticleTitle")
                abstract = article.findtext(".//Abstract/AbstractText")
                journal = article.findtext(".//Journal/Title")
                year = article.findtext(".//PubDate/Year")
                authors = []
                for author in article.findall(".//Author"):
                    last = author.findtext("LastName") or ""
                    first = author.findtext("ForeName") or ""
                    if last or first:
                        authors.append(f"{first} {last}".strip())
                abstracts.append({
                    "pmid": pmid,
                    "title": title,
                    "abstract": abstract,
                    "journal": journal,
                    "year": year,
                    "authors": authors
                })
            except Exception as e:
                print(f"Error parsing article: {e}")
        # Be polite to NCBI servers
        time.sleep(0.5)
    # Save per-topic
    out_path = os.path.join(OUTPUT_DIR, f"pubmed_{topic.replace(' ', '_')}.json")
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(abstracts, f, ensure_ascii=False, indent=2)
    print(f"Saved {len(abstracts)} abstracts for {topic} to {out_path}")
    all_results[topic] = abstracts

print("Done fetching PubMed abstracts.") 