import streamlit as st

st.title("About This Tool")


intro = """
#### Welcome to the Demo of Our AI-Powered Research Tool

This demo showcases our advanced AI tools designed to **elevate due diligence processes** and **uncover actionable growth strategies** with efficiency and precision.

#### Tools Offered in the Demo

1. **Due Diligence Report Generator Tool**  
Empowered by the AI Research Workflow, this tool is built to streamline the creation of Due Diligence reports, enabling you to kickstart the due diligence process for any company in under 2 minutes. \n

2. **Youtube Audience Insight Report Tool**  
This tool leverages advanced analysis of YouTube video data to uncover actionable insights, providing you with a comprehensive understanding of:
   * **Brand Performance**: Analyze audience sentiment and brand perception.
   * **Market & Sub-Industry Trends**: Discover emerging topics and understand consumer behavior.
   * **Competitor Analysis**: Benchmark against competitors and identify strategic opportunities.
"""


st.markdown(intro)

st.write("\n")

st.subheader("Workflow Overview", anchor = False)

st.image("./assets/Beya_AI_Demo.svg", width = 800)

st.image("./assets/Beya_AI_Demo.svg", width = 800)