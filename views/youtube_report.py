import streamlit as st
import pandas as pd
import os
from langchain_openai import ChatOpenAI
import re
import isodate
from googleapiclient.discovery import build
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import Document
import nltk
from nltk.corpus import stopwords
from itertools import cycle
import sys

# ------------------------------------------------------------- #
# --------------------- Environment Setup --------------------- #
# ------------------------------------------------------------- #

# Setup OPENAI APIs
openai_api_keys = [st.secrets['OPENAI_API_KEY1'], st.secrets['OPENAI_API_KEY2']]
openai_api_key_cycle = cycle(openai_api_keys)

def switch_openai_api():
    os.environ["OPENAI_API_KEY"]= next(openai_api_key_cycle)

os.environ["OPENAI_API_KEY"] = next(openai_api_key_cycle)
chatgpt = ChatOpenAI(model="gpt-4o", temperature=0) 

# Prepare stopwords for comment clean 
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Setup Youtube API
youtube_api_keys = [st.secrets['YOUTUBE_API_KEY1'], st.secrets['YOUTUBE_API_KEY2']]
youtube_api_key_cycle = cycle(youtube_api_keys)

youtube_API_KEY = next(youtube_api_key_cycle)
youtube = build("youtube", "v3", developerKey= youtube_API_KEY)

def switch_youtube_api():
    global youtube_API_KEY, youtube
    youtube_API_KEY = next(youtube_api_key_cycle)
    youtube = build("youtube", "v3", developerKey=youtube_API_KEY)
    
    
# Function for simple gpt chat 
def simple_gpt_chat(message):
    try:
        messages = [HumanMessage(content=message)]
        response = chatgpt(messages)
        return response.content
    except Exception as e:
        st.warning("We're sorry, but the OPENAI server encounted error or the request limit has been reached due to too many users at the same time. Please try again after 1 minute, meanwhile you can read the example report..")
        sys.exit()
 
    
    
# ------------------------------------------------------------- #
# ------------------ Youtube Search Functions ----------------- #
# ------------------------------------------------------------- #

# Search for videos for the target company
def search_videos(query, max_results):
    request = youtube.search().list(
        q=query,
        part="id,snippet",
        type="video",
        maxResults=max_results
    )
    response = request.execute()
    videos = []
    for item in response.get("items", []):
        video_id = item["id"]["videoId"]
        title = item["snippet"]["title"]
        channel_id = item["snippet"]["channelId"]
        videos.append({"Video ID": video_id, "Title": title, "Channel ID": channel_id})
    return videos

# Get video details and check if it is a Short
def fetch_video_details(video_id):
    try:
        request = youtube.videos().list(
            part="snippet,statistics,contentDetails",
            id=video_id
        )
        response = request.execute()

        if "items" in response and response["items"]:
            video = response["items"][0]
            snippet = video.get("snippet", {})
            statistics = video.get("statistics", {})
            content_details = video.get("contentDetails", {})

            try:
                duration = isodate.parse_duration(content_details.get("duration", "PT0S")).total_seconds()
            except Exception:
                duration = 0  # Default to 0 seconds if parsing fails

            return {
                "Title": snippet.get("title", "Unknown Title"),
                "Description": snippet.get("description", "No Description Available"),
                "Tags": snippet.get("tags", []),
                "Views": int(statistics.get("viewCount", 0)),
                "Likes": int(statistics.get("likeCount", 0)),
                "Comments Count": int(statistics.get("commentCount", 0)),
                "Duration (seconds)": duration,
                "Is Short": duration <= 60,  # Classify as a Short if duration <= 60 seconds
                "Published At": snippet.get("publishedAt", "Unknown Date"),
            }
        else:
            return {
                "Title": "Unknown Title",
                "Description": "No Description Available",
                "Tags": [],
                "Views": 0,
                "Likes": 0,
                "Comments Count": 0,
                "Duration (seconds)": 0,
                "Is Short": False,
                "Published At": "Unknown Date",
            }

    except Exception as e:
        return {
            "Title": "Unknown Title",
            "Description": "No Description Available",
            "Tags": [],
            "Views": 0,
            "Likes": 0,
            "Comments Count": 0,
            "Duration (seconds)": 0,
            "Is Short": False,
            "Published At": "Unknown Date",
        }

# Get video creator details
def fetch_creator_details(channel_id):
    try:
        request = youtube.channels().list(
            part="snippet,statistics",
            id=channel_id
        )
        response = request.execute()
        if "items" in response and response["items"]:
            channel = response["items"][0]
            snippet = channel.get("snippet", {})
            statistics = channel.get("statistics", {})
            return {
                "Channel Name": snippet.get("title", "Unknown Channel"),
                "Subscribers": int(statistics.get("subscriberCount", 0)),
            }
        else:
            return {
                "Channel Name": "Unknown Channel",
                "Subscribers": 0,
            }

    except Exception as e:
        return {
            "Channel Name": "Unknown Channel",
            "Subscribers": 0,
        }

# Get comments for a video
def fetch_comments(video_id, max_results):
    try:
        request = youtube.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results
        )
        response = request.execute()

        comments = [
            {"comment": item['snippet']['topLevelComment']['snippet']['textOriginal'],
             "likes": item['snippet']['topLevelComment']['snippet']['likeCount']}
            for item in response.get("items", [])
        ]
        return comments
    except Exception as e:
        return []

# Combined above functions to search for a target company and output video and comment details
def master_youtube_query_to_df(query, max_results=20):
    try:    
        results_data = []
        comments_data = []
        videos = search_videos(query, max_results)
        for video in videos:
            video_id = video["Video ID"]
            channel_id = video["Channel ID"]
            video_details = fetch_video_details(video_id)
            creator_details = fetch_creator_details(channel_id)
            comment_count = video_details.get("Comments Count", 0)
            comments_to_fetch = 5 if comment_count < 50 else 10 if comment_count < 100 else 20
            video_comments = fetch_comments(video_id, max_results=comments_to_fetch)
            for comment in video_comments:
                comments_data.append({
                    "Video_ID": video_id,
                    "Comment": comment["comment"],
                    "Likes": comment["likes"]
                })
            results_data.append({
                "Video_ID": video_id,
                "Title": video_details.get("Title"),
                "Description": video_details.get("Description"),
                "Tags": ", ".join(video_details.get("Tags", [])),
                "Views": video_details.get("Views"),
                "Likes": video_details.get("Likes"),
                "Comments_Count": video_details.get("Comments Count"),
                "Duration": video_details.get("Duration (seconds)"),
                "Is_Short": video_details.get("Is Short"),
                "Published_At": video_details.get("Published At"),
                "Channel_Name": creator_details.get("Channel Name"),
                "Subscribers": creator_details.get("Subscribers"),
            })
        video_df = pd.DataFrame(results_data)
        comments_df = pd.DataFrame(comments_data)
        return video_df, comments_df
    except Exception as e:
        st.warning("We're sorry, but the YouTube API daily request limit has been reached. Please try again tomorrow OR read the example report..")
        sys.exit()
        

# Combined above functions to search for a target company and output only video details (for competitor analysis only)
def comp_master_youtube_query(query, max_results=20):
    try:
        results_data = []
        videos = search_videos(query, max_results)
        for video in videos:
            video_id = video["Video ID"]
            channel_id = video["Channel ID"]
            video_details = fetch_video_details(video_id)
            creator_details = fetch_creator_details(channel_id)
            results_data.append({
                "Video_ID": video_id,
                "Title": video_details.get("Title"),
                "Description": video_details.get("Description"),
                "Tags": ", ".join(video_details.get("Tags", [])),
                "Views": video_details.get("Views"),
                "Likes": video_details.get("Likes"),
                "Comments_Count": video_details.get("Comments Count"),
                "Duration": video_details.get("Duration (seconds)"),
                "Is_Short": video_details.get("Is Short"),
                "Published_At": video_details.get("Published At"),
                "Channel_Name": creator_details.get("Channel Name"),
                "Subscribers": creator_details.get("Subscribers"),
            })

        return pd.DataFrame(results_data)
    except Exception as e:
        st.warning("We're sorry, but the YouTube API daily request limit has been reached. Please try again tomorrow OR read the example report..")
        sys.exit()

# ------------------------------------------------------------- #
# ---------------- Process Comments Functions ----------------- #
# ------------------------------------------------------------- #

# Clean comment text to exclude special characters and stopwords
def clean_comment(comment):
    comment = re.sub(r"[^\x00-\x7F]+|[^\w\s.,!?']|[\t\n\r]+", " ", comment)
    comment = " ".join([word for word in comment.split() if word.lower() not in stop_words])
    comment = re.sub(r"\s+", " ", comment).strip()
    return comment

# Given a dataframe, combine all the cleaned comments into chunks of strings
def process_comments(comment_df):
    try:
        comment_df["Cleaned Comment"] = comment_df["Comment"].apply(clean_comment)
        combined_comments = [
            f"{row['Cleaned Comment']}&&{row['Likes']}" if row['Likes'] > 0 else f"{row['Cleaned Comment']}"
            for _, row in comment_df.iterrows()
        ]
        comment_string = "||".join(combined_comments)
        char_limit = 4000 * 4  # Approximate 4 character per token
        chunks = [comment_string[i:i + char_limit] for i in range(0, len(comment_string), char_limit)][:5]
        documents = [Document(page_content=chunk) for chunk in chunks]
        return documents
    except Exception as e:
        return [Document(page_content="")]


# ------------------------------------------------------------- #
# ------------- Target Company Insight Functions -------------- #
# ------------------------------------------------------------- #

# Prompt for target company insight generation
target_company_prompt = ChatPromptTemplate.from_template(
"""You are an expert analyst and professional report writer specializing in due diligence for companies. 
Your goal is to craft a detailed and insightful brand research report for the company: **{target_company}**. 
This report focuses on evaluating the company's branding position and marketing strategy via social media, specifically YouTube.
You are provided with a list of comments extracted from YouTube, separated by "||". At the end of each comment, there may be "&&" followed by a number, which represents the number of likes the comment received.
Treat the likes as an indication of public sentiment: the more likes a comment has, the more people share or agree with the sentiment expressed.
Your task is to analyze these comments and write a professional insight report following the structure and instructions below.

---

Before writing the report, evaluate the provided comments (do not include your evaluation result in the report):
1. **Relevance Check**:
   - Determine if the there are enough relevant comments to **{target_company}** or broader market trends.
   - If many comments are unrelated or mixed with irrelevant topics (e.g., searching for “Apple” and receiving mixed videos unrelated to the company Apple), provide a recommendation suggesting a more specific company name or keyword (e.g., “Apple Inc.” or “Apple technology”).

2. **Sufficiency Check**:
   - Evaluate whether there are enough relevant comments to derive meaningful insights.
   - Only write an "insufficient data report" as a last resort if:
     - Less then 20 meaningful and relevant Comments are find.
     - There is insufficient data for you to write the any insight.
   - In this case:
     - Write an honest summary of findings.
     - State observations like:
       - “Not much video/comments are associated with this company, which might indicate an opportunity for the company to increase its marketing presence on YouTube.”
       - “The provided comments are mixed with unrelated topics, suggesting that the search query could be refined for better results (e.g., using a more specific company name or keyword).”

Try your best to generate insights first and proceed to write the report using the structure and instructions below. 
You should only consider not to follow the below structure as a **last resort**.


### **[Report Outline]**

#### **1. Performance Summary**
- Length: 150-200 words.
- Title: Use `#### Performance Summary` in Markdown format.
- Structure: Include 3-4 bullet points.
  - Each bullet point should start with a very short summary statement in **bold** , followed by the detailed explanation.
- Topics to consider (but not limited to):
  - Brand sentiment trends over time.
  - Brand perception compared to competitors.
  - Trustworthiness and public image.
  - Fandom and customer loyalty.
  - Advocacy for the brand among peers.
  - Brand awareness and identity.
  - Audience insights, including demographics and psychographics of reviewers.

#### **2. Key Topics and Themes**
- Length: 300-500 words.
- Title: Use '#### Key Topics and Themes' in Markdown format.
- Structure:
  - Choose 4-6 aspects from the list below, based on the company's characteristics and the main topics from the comments.
  - For each aspect:
    - State the aspect using '###### **bold**' in Markdown format.
    - Provide 3-4 bullet points of insights, each bullet point should start with a very short summary statement in **bold** 
    - If you are finding a trend in sentiments, give estimatation of the proportion of positive and negative comments (e.g., ~60% reviews showed positive sentiment).
    - After the insight, quote specific comments if necessary:
      - Quote only relevant sentences.
      - Include the number of likes if available (e.g., `"High-quality gymwear is a growing trend" - 45 likes`).

**Aspects to Choose From:**
- **Product Quality**
- **Product Pricing**
- **Common Complaints and Feedback**
- **Sustainability and Ethics**:
  - Topics: Sustainability Practices, Corporate Social Responsibility, Labor Practices.
- **Customer Experience**:
  - Topics: Shopping Experience, Customer Service, Delivery and Shipping.
- **Functional Aspects**:
  - Topics: Ease of Use, Technical Issues, Compatibility and Integration.

#### **3. Actionable Insights**
- Length: 50 - 100 words.
- Title: Use `#### Actionable Insights` in Markdown format.
- Structure: Include 3-4 bullet points.
  - Each bullet point should start with a very short summary statement in **bold** , followed by the detailed explanation.
- Include clear recommendations for (not limited to):
  - Branding and marketing strategies.
  - Areas of improvement (e.g., customer service, product features).
  - Opportunities for differentiation from competitors.

---

### **[General Writing Instructions]**
- Use professional, clear, and straightforward language.
- Avoid excessive complexity or unnecessary jargon.
- Numbers should be rounded and presented without decimals.
- Stay truth to your findings and do not make things up.
- Ensure you are using '####' for titles and you only have "Performance Summary", "Key Topics and Themes", and "Actionable Insights" as titles.


---

### **[Comments Provided for Writing]**
Use the list of comments provided below to craft the report. Each comment may include "&&" followed by a number indicating the number of likes. 
Prioritize comments with higher likes when quoting or drawing insights.
{comments}
"""
)

# Create chain for target company insight generation
chain = create_stuff_documents_chain(
    chatgpt,
    prompt=target_company_prompt,
    document_variable_name="comments" 
)


# ------------------------------------------------------------- #
# -----------------  Market Insight Functions ----------------- #
# ------------------------------------------------------------- #

# Get hashtags from video title and description, along with the original tags
def process_target_hashtags(target_master_df):
    def extract_hashtags(text):
        if isinstance(text, str): 
            return re.findall(r"#\w+", text)
        return []
    try: 
        target_master_df["Title Hashtags"] = target_master_df["Title"].apply(extract_hashtags)
        target_master_df["Description Hashtags"] = target_master_df["Description"].apply(extract_hashtags)
        target_master_df["Tags"] = target_master_df["Tags"].apply(lambda x: x if isinstance(x, list) else [])
        target_master_df["Combined Tags"] = target_master_df.apply(
            lambda row: list(set(row["Title Hashtags"] + row["Description Hashtags"] + row["Tags"])),
            axis=1,
        )
        all_target_hashtags = " ".join(
            [" ".join(hashtags).strip() for hashtags in target_master_df["Combined Tags"] if hashtags]
        ).strip()

        return all_target_hashtags
    except Exception as e:
        return "no hashtags are provided"

# Prompt for keywords to search for market insight generation
get_hashtag_prompt="""You are an analyst specializing in private equity due diligence, focusing on the consumer industry. 
Your role is to assess companies from a branding, marketing, and competitive positioning perspective, with an emphasis on social media platform YouTube.
You are researching the company "{target_company}" to understand its branding strategy, market position, and audience reach via YouTube. 
Your goal is to perform a market analysis by identifying relevant industry trends, topics, and audience behaviors that can help measure the company’s popularity and penetration rate in the broader industry.
Note: You have already searched for videos using the company's name as the query. Your focus should now be on identifying broader industry trends or topics that would help understand the company's target audience and strategic next steps. 
For example, if the target company is "Nike," a broader search term might be "sports apparel."
**If the company name has multiple meanings, assume it refers to a consumer-facing brand relevant to this context**.
You are provided with a list of hashtags from videos related to the target company {target_company_tags}. These hashtags represent the topics and tags frequently associated with the brand and its competitors. 
If the provided hashtags are insightful, consider them when generating search terms. Otherwise, generate your own relevant and insightful keywords or phrases.
**Please carefully review the provided list of hashtags, especially in cases where the company name has multiple meanings. If you find that the hashtags or search results may refer to something unrelated to the intended target, interpret {target_company} as a consumer-facing brand in the context of due diligence research. Use this understanding to generate your own relevant and insightful keywords or phrases that accurately reflect the brand, its market, and industry trends.**
Return **exactly 3 search terms** that would be helpful for this analysis, separated by a single comma (","). Exclude any additional characters, formatting, or commentary. 
Do not include hashtags ("#") in your response and ensure the search terms do not contain commas.
"""



# For the top 3 keywords for market trend, get relevant video and comments
def fetch_market_data(broad_search, max_results_per_term=7):
    try:
        searchterms = broad_search.split(", ")
        market_video_data = pd.DataFrame()
        market_comments_data = pd.DataFrame()   
        for term in searchterms:
            video_df, comments_df = master_youtube_query_to_df(term, max_results=max_results_per_term)
            market_video_data = pd.concat([market_video_data, video_df], ignore_index=True)
            market_comments_data = pd.concat([market_comments_data, comments_df], ignore_index=True)
        
        return market_video_data, market_comments_data
    except Exception as e:
        st.warning("We're sorry, but the YouTube API daily request limit has been reached. Please try again tomorrow OR read the example report..")
        sys.exit()

# Prompt for market insight generation
market_prompt = ChatPromptTemplate.from_template(
"""You are an expert analyst and professional report writer specializing in due diligence for companies. Your goal is to craft a detailed and insightful market research report for the company: **{target_company}**. This report focuses on identifying broader market trends and analyzing the target audience group to assess the company’s strengths, weaknesses, and strategic opportunities.

You are provided with:
1. **Insights for the company**: This includes specific findings about {target_company}'s branding, marketing, and audience from a prior analysis.
2. **Comments for the general industry and trends**: These comments are separated by "||", with each comment optionally followed by "&&" and a number indicating the number of likes. Treat the likes as an indication of public sentiment: the more likes a comment has, the more widely it resonates with the audience.

---

### **[Report Outline]**

#### **1. Market Trend**
- Length: 300-400 words.
- Title: Use `#### Key Market Trend` in Markdown format.
- Structure:
  - Identify **3-4 key market trends or topics** based on the comments.
  - For each trend:
    - Provide a **summary** of what the trend is about, including its relevance to the industry and audience.
    - Offer **details and insights** drawn from the comments, quoting key phrases if necessary:
      - Quote only relevant sentences.
      - Include the number of likes for each quoted comment (e.g., `"High-quality gymwear is a growing trend" – 45 likes`).
    - Highlight how {target_company} is performing in relation to this trend:
      - Mention strengths, weaknesses, or areas where the company aligns with or deviates from the trend.

#### **2. Market Overview**
- Length: 200-300 words.
- Title: Use `#### Market Overview` in Markdown format.
- Structure:
  - Describe the **current position of {target_company} in the market**, including comparisons to competitors.
  - Summarize **industry trends and dynamics**, focusing on the larger forces shaping the market.
  - Provide a **competitor analysis**:
    - Highlight {target_company}'s key competitors and their strengths in addressing these trends.
    - Identify gaps or opportunities where {target_company} could differentiate itself.

#### **3. Recommendations**
- Length: 150-200 words.
- Title: Use `#### Recommendations for {target_company}` in Markdown format.
- Structure:
  - Provide **3 actionable recommendations** for how {target_company} can leverage the market trends to improve its branding and marketing strategy.
  - Ensure recommendations are specific, realistic, and actionable.
  - Example recommendation structure:
    - **What to do**: e.g., Focus on sustainability messaging.
    - **Why it matters**: e.g., Sustainability resonates strongly with the target audience (~70% positive sentiment in comments about eco-friendly practices).

---

### **[General Writing Instructions]**
- Use professional, clear, and concise language.
- Stay truth to your findings and do not make things up.
- Numbers should be rounded and presented without decimals.
- Ensure the analysis is meaningful and actionable, if less then 20 meaningful and relevant comments are find, you can make the overall section short and observations like:
       - “Not much video/comments are associated with this company, which might indicate an opportunity for the company to increase its marketing presence on YouTube.”
       - “The provided comments are mixed with unrelated topics, suggesting that the search query could be refined for better results (e.g., using a more specific company name or keyword).”
- Ensure you are using '####' for titles and you only have "Key Market Trend", "Market Overview", and "Recommendations" as titles.

---

### **[Provided Data for Analysis]**
- **Insights from the comments for {target_company}**: {target_company_insight}
- **Comments for overall market**: {comments}



"""
)

# Create chain for market insight generation
market_chain = create_stuff_documents_chain(
    chatgpt,
    prompt=market_prompt,
    document_variable_name="comments" 
)


# ------------------------------------------------------------- #
# --------------- Competitor Analysis Functions --------------- #
# ------------------------------------------------------------- #

# Prompt for getting top competitors
get_competitor_prompt="""You are an analyst specializing in private equity due diligence, with a focus on the consumer industry. 
Your role is to assess companies from a branding, marketing, and competitive positioning perspective.
You are researching the company "{target_company}" to understand its branding strategy, market position, and potential threats from competitors. 
If the company name has multiple meanings, assume it refers to a consumer-facing brand relevant to this context.
Perform a competitive analysis and identify the **top 2 competitor companies** that pose the biggest threat to the target company and are excelling in marketing. 
Return your response as **exactly two brand names separated by a single comma (",")**, with no additional characters, commentary, or formatting. 
Ensure the brand names do not contain commas and ensure **your response sentence only contains one comma**
"""


# For each company, calculate the key metrics and output a table

def calculate_youtube_metrics(df, Company, category):
    def format_number(num):
        if num >= 1_000_000:
            return f"{num / 1_000_000:.1f}M"
        elif num >= 1_000:
            return f"{num / 1_000:.1f}K"
        return str(round(num))
    
    total_views = df["Views"].sum()
    avg_views_per_video = df["Views"].mean()
    total_likes = df["Likes"].sum()
    likes_to_views_ratio = (df["Likes"] / df["Views"]).mean() * 100
    total_comments = df["Comments_Count"].sum()
    avg_comments_per_video = df["Comments_Count"].mean()
    comments_to_views_ratio = (df["Comments_Count"] / df["Views"]).mean() * 100
    shorts_percentage = df["Is_Short"].mean() * 100
    avg_subscribers = df["Subscribers"].mean()
    
    return {
        "Company" : Company,
        "Category": category,
        "Total Views": format_number(total_views),
        "Avg Views per Video": format_number(avg_views_per_video),
        "Total Likes": format_number(total_likes),
        "Likes-to-Views Ratio (%)": f"{round(likes_to_views_ratio)}%",
        "Total Comments": format_number(total_comments),
        "Avg Comments per Video": format_number(avg_comments_per_video),
        "Comments-to-Views Ratio (%)": f"{round(comments_to_views_ratio,2)}%",
        "Shorts (%)": f"{round(shorts_percentage)}%",
        "Avg Subscribers": format_number(avg_subscribers),
    }


    




# Prompt for getting metric analysis insights
summary_insight_prompt="""You are an expert data analyst specializing in marketing, branding, and audience analysis. 
Your goal is to analyze YouTube performance data to provide actionable insights for the company: **{target_company}**. 
The analysis focuses on comparing the target company’s YouTube metrics to those of its top 2 competitors to evaluate audience size, engagement quality, and strategic opportunities.

Use the provided summary table to identify patterns, trends, and opportunities. 
Follow the structure below, but feel free to expand your analysis with additional observations based on the data.

---

### **[Provided Data]**
Below is a summary table of YouTube metrics for the target company (first row) and its top 2 competitors (second and third rows). The columns include:
- **Total Views**: Total views across analyzed videos.
- **Average Views per Video**: The average views for each video.
- **Total Likes**: Total likes received across analyzed videos.
- **Likes-to-Views Ratio (%)**: Average percentage of likes relative to views.
- **Total Comments**: Total comments across analyzed videos.
- **Comments-to-Views Ratio (%)**: Average percentage of comments relative to views.
- **Shorts (%)**: Percentage of analyzed videos that are YouTube Shorts.
- **Average Subscribers**: Average number of subscribers for the creators of the analyzed videos.

Table: {summary_markdown}
---

### **Analysis Aspects**

Use the following goals as a guide for your analysis:

#### **1. Audience Size and Market Reach**
- Compare **Total Views** and **Average Subscribers** across the target company and competitors.  
- Identify the company with the largest audience size and what this indicates about market reach and brand awareness.  
- Discuss opportunities for the target company to expand its audience size.

#### **2. Engagement Quality**
- Evaluate **Likes-to-Views Ratio (%)** and **Comments-to-Views Ratio (%)**:  
  - Identify which company demonstrates stronger engagement rates.  
  - Discuss how these metrics reflect audience loyalty, brand affinity, and marketing effectiveness.

#### **3. Shorts Strategy**
- Analyze the **Shorts (%)** column to determine which company relies most on Shorts.  
- Discuss how this aligns with YouTube’s audience trends and whether the target company could benefit from changing its Shorts strategy.

#### **4. Competitive Insights**
- Compare and contrast the performance of the target company against its competitors across all metrics.  
- Highlight areas where the target company outperforms or lags behind and suggest strategic actions.

#### **5. Additional Observations**
- Look for interesting patterns, correlations, or anomalies in the data that might provide deeper insights.  
- For example:  
  - Are there outliers in the **Average Views per Video** or **Total Comments**?  
  - Does the use of Shorts correlate with higher or lower engagement rates?  
  - Are there potential gaps in content strategy that could be leveraged?

---

### **[Report Format]**

1. **Key Insights**:
   - Genrate 4-5 indepth analysis insights based on the data.  
   - Use comparisons between the target company and competitors to support your findings.  
2. **Opportunities and Recommendations**:
   - Suggest strategic actions the target company can take to improve audience size, engagement, or market reach.  
   - Highlight areas where the company could differentiate itself from competitors.

---

### **[General Writing Instructions]**
- Write in clear, professional, and data-driven language.
- Ensure the analysis is well-structured and focused on meaningful insights.
- Use rounded numbers without decimals for consistency.
- If you identify data gaps or inconsistencies, mention them explicitly and discuss their potential impact on the analysis.
- Keep the tone analytical and insightful.

---


"""



# ------------------------------------------------------------- #
# --------------- Streamlit workflow generation --------------- #
# ------------------------------------------------------------- #

st.title("Youtube Audience Insight Report")

ytb_lululemon = f"https://github.com/Virhahahaha/AI_Tool/blob/main/sample_reports/youtube_lululemon.pdf"
ytb_ouraring = f"https://github.com/Virhahahaha/AI_Tool/blob/main/sample_reports/youtube_oura.pdf"
ytb_parachute = f"https://github.com/Virhahahaha/AI_Tool/blob/main/sample_reports/youtube_parachute.pdf"
ytb_allbirds = f"https://github.com/Virhahahaha/AI_Tool/blob/main/sample_reports/youtube_allbirds.pdf"
ytb_olipop = f"https://github.com/Virhahahaha/AI_Tool/blob/main/sample_reports/youtube_olipop.pdf"

st.markdown(f"""
##### Welcome to the Youtube Audience Insight Report Generator!

This tool leverages advanced analysis of YouTube video data to uncover actionable insights, providing you with a comprehensive understanding of:
- **Brand Performance**: Analyze audience sentiment and brand perception.
- **Market & Sub-Industry Trends**: Discover emerging topics and understand consumer behavior.
- **Competitor Analysis**: Benchmark against competitors and identify strategic opportunities. \n
Before you get started, make sure to check out the "**About This**" page for an overview (accessible via the left-side navigation panel). \n

**Please Note**: This prototype currently utilizes a personal API with a daily usage limit. If you receive a warning indicating that the API limit has been reached, we kindly ask you to try again the following day. \n
In the meantime, feel free to explore our sample reports for inspiration and insights using the links below: \n
[Lululemon]({ytb_lululemon}) | [Oura Ring]({ytb_ouraring}) | [Parachute]({ytb_parachute}) | [Allbirds]({ytb_allbirds}) | [Olipop]({ytb_olipop})
""", unsafe_allow_html=True)

st.divider()

st.markdown("##### Generate your own report:")

# Get target company from user input
target_company_input = st.text_input("Enter your target company and press enter key: (Example: L Catterton )")

if target_company_input:
    target_company = target_company_input.title()
    st.write(f"You Entered: {target_company}")
    with st.spinner("Searching Youtube Content... Please wait!"):
        
        # Search for youtube video and comments and format as dfs
        target_master_df, target_comment_df  = master_youtube_query_to_df(target_company)
        
        # Clean and process all the comments into langchain documents
        target_comment_documents = process_comments(target_comment_df)
        
        # Switch api to avoid token limit within 1 minute
        switch_openai_api()
        
        # GPT to summarize the documents and return insights for target company
        try:    
            target_company_insights = chain.invoke({"comments" : target_comment_documents,"target_company" : target_company})
        except Exception as e:
            st.warning("We're sorry, but the OPENAI server encounted error or the request limit has been reached due to too many users at the same time. Please try again after 1 minute, meanwhile you can read the example report..")
            sys.exit()
            
        # Present the target company analysis insights
        st.divider()
        st.markdown("""
            <h3 style='color:darkblue;'>Brand Insight Report</h3>
                    """,
            unsafe_allow_html=True
        )
        st.write(target_company_insights)
        st.divider()
        st.markdown("""
            <h3 style='color:darkblue;'>Market and Audience Insight Report</h3>
                    """,
            unsafe_allow_html=True
        )         
        
        # GPT to generate keywords to search to understand the market and trend related to the target company based on video hashtags
        broad_search = simple_gpt_chat(get_hashtag_prompt.format(target_company = target_company, target_company_tags = process_target_hashtags(target_master_df) ))

        # Search for youtube video and comments for those keywords
        switch_youtube_api()
        youtube = build("youtube", "v3", developerKey=youtube_API_KEY)
        market_video_data, market_comments_data = fetch_market_data(broad_search)
        
       # Clean and process all the comments into langchain documents
        market_comment_documents = process_comments(market_comments_data)
        
        # Switch api to avoid token limit per minute
        switch_openai_api()
        
        # GPT to summarize the documents and return insights for market trend insights
        try:
            market_insights = market_chain.invoke({"comments" : market_comment_documents,"target_company" : target_company, "target_company_insight" : target_company_insights} )
        except Exception as e:
            st.warning("We're sorry, but the OPENAI server encounted error or the request limit has been reached due to too many users at the same time. Please try again after 1 minute, meanwhile you can read the example report..")
            sys.exit()
            
        # Present the market trend analysis insights
        st.write(market_insights) 
        st.divider()
        st.markdown("""
            <h3 style='color:darkblue;'>Competitor Insight Report</h3>
                    """,
            unsafe_allow_html=True
        )
                
        # GPT to generate competitor companies
        competitor_company = simple_gpt_chat(get_competitor_prompt.format(target_company = target_company))
        comp1,comp2= competitor_company.split(",")
        
        # Get Competitor youtube video information
        comp1_df = comp_master_youtube_query(comp1)
        switch_youtube_api()
        youtube = build("youtube", "v3", developerKey= youtube_API_KEY)
        comp2_df = comp_master_youtube_query(comp2)
        
        # Summarize the key video performance metrics for the target company and it's competitors
        target_metrics = calculate_youtube_metrics(target_master_df, target_company, "Target Company")
        comp1_metrics = calculate_youtube_metrics(comp1_df, comp1, "Competitor Company")
        comp2_metrics = calculate_youtube_metrics(comp2_df, comp2, "Competitor Company")               
        summary_df = pd.DataFrame([target_metrics, comp1_metrics, comp2_metrics])
        summary_markdown = summary_df.to_markdown(index=False)
        
        # GPT to generate competitor analysis insights based on the key video performance metrics
        competitor_insights = simple_gpt_chat(summary_insight_prompt.format(summary_markdown = summary_markdown,target_company = target_company ))
        
        # Present the competitor analysis insights
        st.write(competitor_insights) 
        
        st.markdown("##### Detailed Metrics:")
        styled_df = summary_df.style.set_properties(**{
            'text-align': 'center',
            'font-size': '12px',  
            'padding': '2px'  
        }).set_table_styles([
            {"selector": "th", "props": [("text-align", "center"), ("font-size", "13px"), ("background-color", "#f4f4f4")]},  
            {"selector": "td", "props": [("border", "1px solid #ddd"), ("padding", "4px")]}, 
            {"selector": "table", "props": [("border-collapse", "collapse"), ("width", "80%"), ("margin", "auto")]}  
        ])
        st.markdown(styled_df.hide(axis="index").to_html(), unsafe_allow_html=True)







