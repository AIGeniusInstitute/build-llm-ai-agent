import json
import os
from pprint import pprint
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

tavily_api_key = os.environ["TAVILY_API_KEY"]


def search_by_tavily(q) -> str:

    url = "https://api.tavily.com/search"

    payload = {
        "query": q,
        "topic": "general",
        "search_depth": "basic",
        "chunks_per_source": 3,
        "max_results": 2,
        "time_range": None,
        "days": 7,
        "include_answer": True,
        "include_raw_content": True,
        "include_images": False,
        "include_image_descriptions": False,
        "include_domains": [],
        "exclude_domains": [],
        "country": None
    }
    headers = {
        "Authorization": f"Bearer {tavily_api_key}",
        "Content-Type": "application/json"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    return response.text



if __name__ == '__main__':
    q = 'Model Context Protocol'

    # results = search_by_serpapi(q)
    results = search_by_tavily(q)

    pprint(results)
