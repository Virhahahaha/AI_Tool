from __future__ import annotations as _annotations
from IPython.display import display, Markdown
from typing import List
from pydantic import BaseModel, Field
import os
from tavily import TavilyClient
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import streamlit as st
import re
from itertools import cycle

# ------------------------------------------------------------- #
# --------------------- Environment Setup --------------------- #
# ------------------------------------------------------------- #

# Setup OPENAI APIs
openai_api_keys = [st.secrets['OPENAI_API_KEY1'],st.secrets['OPENAI_API_KEY2'],st.secrets['OPENAI_API_KEY3'],st.secrets['OPENAI_API_KEY4'],st.secrets['OPENAI_API_KEY5']]
openai_api_key_cycle = cycle(openai_api_keys)

def switch_openai_api():
    os.environ["OPENAI_API_KEY"]= next(openai_api_key_cycle)
    
os.environ["OPENAI_API_KEY"] = next(openai_api_key_cycle)

chatgpt = ChatOpenAI(model="gpt-4o", temperature=0) 

# Initialize Tavily Clients
tavily_client = TavilyClient(api_key=st.secrets['TAVILY_API'])

# Pydantic Models for Structured Output
class SearchQuery(BaseModel):
    search_query: str = Field(None, description="Search queries")
    
class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="Search queries",
    )
    
Queries.model_rebuild()

# ------------------------------------------------------------- #
# ------------------ Tavily Search Functions ------------------ #
# ------------------------------------------------------------- #

# Prompt for generating search queries
get_search_queries_prompt="""You are a skilled due diligence analyst responsible for conducting in-depth research on consumer facing companies to support decision-making processes for a private equity firm. 
Your current task is to generate a set of highly targeted and effective web search queries to assist in writing a **due diligence report** for the company: **{target_company}**.

The specific focus of this research is the following topic: **{report_section}**.

Your goal is to craft **{number_of_queries}** detailed search queries that will surface comprehensive and high-quality information relevant to the assigned report section.

**If the company name has multiple meanings, assume it refers to a consumer-facing brand within the context of this report, ensure that your queries are narrowly tailored to surface results specifically relevant to this brand, avoiding unrelated or ambiguous content.**
---

### **Instructions for Crafting Search Queries**

#### **Key Objectives**
1. Ensure that the search queries align with the **key topic**: **{report_section}**.
2. Focus on uncovering diverse perspectives and reliable information sources.
3. Aim for recent and authoritative results, incorporating year markers where appropriate (e.g., "2024").

#### **Search Query Guidelines**
1. **Specificity**:
   - Avoid generic terms; tailor each query to the topic for precision.
   - Include terms directly related to the topic, such as technical terminology, key concepts, or business-specific language.

2. **Relevance to Timeframe**:
   - Ensure queries prioritize recent information by adding year markers (e.g., "2024 trends").
   - Include phrases like "latest insights" or "recent developments" when appropriate.

3. **Diversity of Aspects**:
   - Cover a broad range of relevant subtopics related to the report section.
   - Think about different dimensions of the topic, such as:
     - **Market trends**: "Emerging technologies in sportswear (2024)"
     - **Competitor comparisons**: "Lululemon vs Nike brand perception"
     - **Operational strategies**: "Supply chain practices of Lululemon"

4. **Source Authority**:
   - Target trusted and authoritative sources such as:
     - **Official websites and documentation**: "Company white papers"
     - **Industry-specific blogs and publications**: "Market analysis reports"
     - **Academic research**: "University studies on customer behavior"

---

### **Expected Output**
You should generate **{number_of_queries}** search queries that:
1. Are highly specific and tailored to the topic.
2. Balance technical depth with accessibility.
3. Reflect diverse subtopics and perspectives.
4. Target authoritative and credible sources.

For example:
- **Good Query**: "Sustainability practices in the fashion industry (2024) - Corporate Social Responsibility of Lululemon"
- **Bad Query**: "Lululemon news"

---

### **Additional Notes**
- Use a clear and professional tone.
- Avoid overly broad or ambiguous search terms.
- Ensure the queries are actionable and relevant for web searches.

"""


# gpt to generate the search queries about the company and section topic
def get_search_queries(target_company,report_section,number_of_queries):
    try:
        structured_llm = chatgpt.with_structured_output(Queries)
        queries = structured_llm.invoke([HumanMessage(content=get_search_queries_prompt.format(target_company=target_company, report_section=report_section, number_of_queries=number_of_queries))])
        return queries.queries
    except Exception as e:
        st.warning("Error generating search queries. Please try again later.")
        return []

# Function to clean the search result to exclude special characters 
def clean_content(search_results):
    clean_results = []
    char_limit = 1500 * 4
    seen_urls = set()
    
    try:
        if isinstance(search_results, dict):
            result_list = search_results['results']
        elif isinstance(search_results, list):
            result_list = []
            for response in search_results:
                if isinstance(response, dict) and 'results' in response:
                    result_list.extend(response['results'])
                else:
                    result_list.extend(response)
        for i, result in enumerate(result_list, start=1):
            title = result['title']
            url = result['url']
            
            if url in seen_urls:
                continue
            seen_urls.add(url)
            
            content = result['content']
            raw_content = result.get('raw_content', '')
            if raw_content is None:
                raw_content = ''           
            raw_content = re.sub(r"[^\x00-\x7F]+|[^\w\s.,!?']|[\t\n\r]+", " ", raw_content)
            raw_content_trimmed = raw_content[:char_limit] if len(raw_content) > char_limit else raw_content
            raw_content_cleaned = " ".join(raw_content_trimmed.split())

            clean_results.append(
                f"Result {i}:\n"
                f"Title: {title}\n"
                f"URL: {url}\n"
                f"Summary: {content}\n"
                f"Raw Content (trimmed): {raw_content_cleaned}\n"
            )
        return "\n".join(clean_results).strip()
    except Exception as e:
        return "Error processing this result."


# Let Tavily to research based on the queries and get the content

def tavily_search(queries_to_search,tavily_topic): 
    try:  
        search_results = []
        for query in queries_to_search:
            result = tavily_client.search(
                query.search_query,
                max_results=3,
                include_raw_content=True,
                topic=tavily_topic,
                days=365,
                include_images=False,
            )
            search_results.append(result)
        source_str = clean_content(search_results)
        return source_str
    except Exception as e:
        return "Error processing this result."    

# ------------------------------------------------------------- #
# ---------------- Writing Prompt and Function ---------------- #
# ------------------------------------------------------------- #

# Prompt for writing the busines profile section
business_profile_prompt = """You are an expert due diligence analyst and professional report writer specializing in crafting concise and insightful business profiles for private equity purposes.
You are tasked with generating a quick business profile summary for the company: **{target_company}**. Your analysis should focus on synthesizing key web search content to produce a professional and clear summary.
If the company name has multiple meanings, assume it refers to a consumer-facing brand relevant to this context.
The profile must include:
- A brief history of the company, including its founding and core mission or values.
- A summary of what the company offers, highlighting any unique or proprietary products or services.
- Additional notable or unique features that distinguish the company in its industry.
Your response must adhere to the following requirements:
- Write in simple, clear, and professional language.
- Keep the summary strictly under 150 words.
- Use `### Business Profile` as the title in Markdown format.
- Provide the summary as a single paragraph with no additional commentary or formatting beyond what is specified.
"""


# Prompt for writing the general section
section_aspects_prompt = """You are an expert due diligence analyst and professional report writer specializing in consumer-facing private equity. 
Your task is to craft a concise, data-driven, and actionable report section for the company: **{target_company}**, focused specifically on the aspct of: **{report_section}**.

The report will be part of a Confidential Information Memorandum (CIM) and must meet the highest professional standards. Use the structure, writing guidelines, and quality checks outlined below.

---

### **[Report Structure]**

- **Length:** 100–150 words.
- **Purpose:** Deliver 1–3 key insights related to **{report_section}**.
- Structure:
  - Use `### {report_section}` as the title in Markdown format.
  - For each insight:
    - Begin with a **bold headline**.
    - Give 2~3 bullet points to explaine the insight, supported by specific findings.
    - Use concrete examples or case studies from the source material (`{context}`) to validate the insight.
    - **Do not include** any source or URL links when writing insights and detailed bullet points.
  - At the end, after the insight, include a line labeled "Sources" to list valid URLs in markdown format:
    - Format the URLs as markdown hyperlinks with descriptive names like [Link1](https://example.com), [Link2](https://example2.com), etc.
    - Ensure the link names are sequentially numbered (e.g., Link1, Link2, Link3) with no skipped or repeated numbers.
    - Do not include raw URLs in the text; instead, use [Link1](URL) syntax for hyperlinks.

---

### **[Writing Guidelines]**

1. **Language and Tone**:
   - Write in clear, professional, and technical language.
   - Avoid marketing or promotional tone.
   - Prioritize precision and factual accuracy.

2. **Formatting**:
   - Use Markdown for titles, key insights, and bullet points.
   - Ensure the section is well-organized, with clear headings and subheadings.
   - Ensure the response is free of formatting errors or broken text, and verify the clarity of all sentences.

3. **Word Limit**:
   - Strictly limit the entire section to 200 words (excluding source citations).

4. **Focus**:
   - Stay specific to **{report_section}** for **{target_company}**.
   - Ensure insights are relevant.

5. **Examples and Data**:
   - Include at least one specific example or case study to support the findings.
   - Use concrete metrics, benchmarks, or quotes from the source material.

6. **Ensure your report adheres to the following Structure**:
   - Ensure you have 1~3 Key Insights sections, and for each Key Insights section, ensure you start with a **bold** title and followed by 2~3 bullet points
   - Uses proper Markdown formatting and clear, concise language.
   - ** Do not mention URLs or source links when writing any insights and bullet points, have a seperate source line at the end**
   - **For source line, double check to ensure you are using descriptive and sequentially numbered names (e.g., Link1, Link2, Link3) with no skipped or repeated numbers**

---

### **[Source Material for Analysis]**
Use the provided context to craft your analysis:  
**{context}**
If there is no context for the analysis, still follow the report structure and State observations like:
       - “Not enough research materials are found on this topic, try increase the number of searches to gather more insights.”
 

"""


# Prompt for writing the summary section
key_summary_prompt="""You are an expert technical writer specializing in crafting concise and impactful executive summaries for due diligence reports. 
Your task is to synthesize information from the available report content to create the **Key Insights Summary** for the company: **{target_company}**.

The purpose of this summary is to distill the core insights, motivations, and conclusions from the report into a clear and compelling narrative.
The report will be part of a Confidential Information Memorandum (CIM) and must meet the highest professional standards. Use the structure, writing guidelines, and quality checks outlined below.

---

### **[Guidelines for Writing]**

#### **1. Writing Objective**
- Focus on crafting a narrative that:
  - Summarizes the key findings and themes from the report.
  - Provides an overarching perspective on the company’s position, strengths, weaknesses, and opportunities.

#### **2. Structure and Style**
- Use `### Key Insights Summary` as the title in Markdown format.
- Write in **simple, professional, and clear language**.
- Limit the summary to **100–150 words**.
- Focus on a **narrative arc**:
  - **Introduction**: Briefly introduce the purpose of the report and its focus areas.
  - **Core Insights**: Highlight the most critical findings, weaving them into a cohesive story.
  - **Conclusion**: Provide a succinct conclusion or a forward-looking statement.
  

#### **3. Formatting and Tone**
- Do not use lists, tables, or structural elements within the summary.
- Write in paragraph form with a smooth flow between ideas.
- Avoid jargon or overly technical language—prioritize clarity and impact.
- Ensure the response is free of formatting errors or broken text, and verify the clarity of all sentences.

---

### **[Source Material for Synthesis]**
Use the following report content to craft your summary:  
**{context}**

---

### **[Quality Checks]**
Ensure the Key Insights Summary adheres to the following standards:
1. **Relevance**:
   - Accurately reflects the main findings and themes from the report content.
   - Focuses on insights that align with the report’s purpose and the target company’s strategic priorities.
2. **Narrative Flow**:
   - Creates a clear and engaging narrative arc that connects the introduction, insights, and conclusion.
3. **Length**:
   - Strictly between 100–150 words.
4. **Clarity**:
   - Uses simple, professional, and error-free language.
   - Avoids unnecessary complexity or irrelevant details.

---
"""



sections_to_write = ["Business Profile",
                    "Products/Services review",
                    "Market and Competitive Landscape",
                    "Risk and Challenges",
                    "Financial information and valuation",
                    "News Trend"]


st.title("Due Diligence Report")

dd_lululemon = f"https://github.com/Virhahahaha/AI_Tool/blob/main/sample_reports/dd_lululemon.pdf"
dd_ouraring = f"https://github.com/Virhahahaha/AI_Tool/blob/main/sample_reports/dd_oura.pdf"
dd_parachute = f"https://github.com/Virhahahaha/AI_Tool/blob/main/sample_reports/dd_parachute.pdf"
dd_allbirds = f"https://github.com/Virhahahaha/AI_Tool/blob/main/sample_reports/dd_allbirds.pdf"
dd_olipop = f"https://github.com/Virhahahaha/AI_Tool/blob/main/sample_reports/dd_olipop.pdf"

st.markdown(f"""
##### Welcome to the AI-Powered Due Dilligence Report Generator!
Empowered by the AI Research Workflow, this tool is built to streamline the creation of Due Diligence reports, enabling you to kickstart the due diligence process for any company in under 2 minutes. \n
Before you get started, make sure to check out the "**About This**" page for an overview (accessible via the left-side navigation panel). \n
Explore sample reports for inspiration and insights using the links below: \n
[Lululemon]({dd_lululemon}) | [Oura Ring]({dd_ouraring}) | [Parachute]({dd_parachute}) | [Allbirds]({dd_allbirds}) | [Olipop]({dd_olipop})
""", unsafe_allow_html=True)

st.divider()

st.markdown("##### Generate your own report:")
            
# Input for the target company
target_company_input = st.text_input("Enter your target company and press enter key: (Example: L Catterton )")

if target_company_input:
    target_company = target_company_input.title()
    st.write(f"You Entered: {target_company}")
    with st.spinner("AI Research Agent Working... Please wait!"):
        try: 
            partial_report = ""
            for report_section in sections_to_write:
                      
                 # Determine the Tavily topic
                tavily_topic = "news" if report_section == "News Trend" else "general"
                
                # Generate search queries and fetch Tavily search results
                queries_to_search = get_search_queries(target_company,report_section,2)
                search_result = tavily_search(queries_to_search, tavily_topic)
                
                # Switch api to avoid token limit within 1 minute
                switch_openai_api()
                
                # Generate section content using GPT
                if report_section == "Business Profile":
                    section_content = chatgpt.invoke([
                        HumanMessage(content=business_profile_prompt.format(target_company=target_company))
                        ]).content
                else:
                    section_content = chatgpt.invoke([
                        HumanMessage(content=section_aspects_prompt.format(target_company=target_company, report_section = report_section, context=search_result))
                        ]).content
                    
                # Append the section content to the partial report
                partial_report += section_content + "\n\n"

            # Switch api to avoid token limit within 1 minute
            switch_openai_api()
                       
            # Generate the summary report    
            summary_report = chatgpt.invoke([
                HumanMessage(content=key_summary_prompt.format(target_company=target_company, context=partial_report))
                ]).content
            
            # Combine to final report  
            final_report = f"{'#' * 2} {target_company} Research Report" + "\n\n" + summary_report + "\n\n" + partial_report
            st.markdown(final_report)
            
        except Exception as e:
            st.warning("We're sorry, but the OPENAI server encounted error or the request limit has been reached due to too many users at the same time. Please try again after 1 minute, meanwhile you can read the example report..")




