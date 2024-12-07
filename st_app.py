from firecrawl import FirecrawlApp
import pandas as pd
import numpy as np
import openai
from googlesearch import search
import streamlit as st


openai.api_key = st.secrets['OPENAI_API_KEY']

def google_news_results(target_company, num_results=10,result_urls=[]):
    search_query = f"{target_company} news -site:{target_company}.com -site:instagram.com -site:reddit.com -site:facebook.com -site:twitter.com -site:linkedin.com -site:yelp.com -site:quora.com -site:wikipedia.org"
    search_results = search(search_query, num_results=num_results)
    for idx, result in enumerate(search_results, 1):
        if "pdf" not in result:
            result_urls.append(result)
    return result_urls


def firecrawl_scrape(url):
    app = FirecrawlApp(api_key=st.secrets['FIRECRAWL_API_KEY'])
    try:
        scraped_data = app.scrape_url(url)
    except Exception as e:
        print(f"Unable to scrape, Exception: {e}")
        return e    
    return scraped_data["markdown"]

def company_profile_prompt(company: str) -> str:
        prompt = (
            f"You are an analytst and your role is to perform due dilligence of companies."
            f"---\n"
            f"Can you briefly summarize on what is the company {company} ?"
            f"---\n"
            f"Please stay true to your findings and do not make things up. Your output should be a paragraph less than 100 words "
        )
        return prompt

def google_news_prompt(company: str,company_profile: str, article_text: str) -> str:
        prompt = (
            f"You are an analytst and your role is to perform due dilligence of companies. "
            f"---\n"
            f"Can you briefly summarize on what is artical about, specifically regarding this {company}? Here is more detail about this company: {company_profile}"
            f"---\n"
            f"Here is the artical detail : {article_text} "
            f"Please stay true to your findings and do not make things up. Your output should be a paragraph less than 100 words"
        )
        return prompt

def google_news_summary_prompt(company: str,company_profile: str, article_summaries: list) -> str:
        prompt = (
            f"You are an analytst and your role is to perform due dilligence of companies. "
            f"---\n"
            f"I have a list of summary of recent articals about this {company} and here is more detail about this company: {company_profile}. Can you output an analysis report based on the recent news about this company."
            f"---\n"
            f"Here is the list of summary of recent articals : {article_summaries} "
            f"Please stay true to your findings and do not make things up. Your output should be a paragraph less than 200 words"
        )
        return prompt

def chat_completion_request(messages):
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  
            messages=[
            {
            "role": "system",
            "content": messages
            }
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Unable to scrape, Exception: {e}")
        return e



st.write("Research Agent")

target_company = st.text_input("Enter your target company: ")

if target_company:
    st.write(f"You entered: {target_company}")
    with st.spinner("Processing Google... Please wait!"):
        company_profile = chat_completion_request(company_profile_prompt(target_company))
        top_results = google_news_results(target_company, num_results=3)
        articals_summary = []
        for urls in top_results:
            artical = firecrawl_scrape(urls)
            chat_result = chat_completion_request(google_news_prompt(target_company,company_profile, artical))
            articals_summary.append(chat_result)
        news_summary = chat_completion_request(google_news_summary_prompt(target_company,company_profile, articals_summary))
    st.write("Target Company Profile")
    st.write(company_profile)
    st.write("Recent News Summary")
    st.write(news_summary)









