import streamlit as st

# --- PAGE SETUP ---

# --- https://fonts.google.com/icons ---

about_this_tool_page = st.Page(
    page = "views/about_this_tool.py",
    title = "About This",
    icon = ":material/home:",
    default = True,
)

tool_1_page = st.Page(
    page = "views/summary.py",
    title = "Due Diligence Report",
    icon = ":material/book_4_spark:",
)

tool_2_page = st.Page(
    page = "views/youtube_report.py",
    title = "Youtube Audience Insight Report",
    icon = ":material/youtube_activity:",
)

pg = st.navigation(
    {
        "About This Tool": [about_this_tool_page],
        "Tools": [tool_1_page,tool_2_page]
    }
)

#st.logo("assets/logo.png")
st.sidebar.text("Demo Made By Beya")


st.markdown(
    """
    <style>
    footer {visibility: hidden;}
    </style>
    """,
    unsafe_allow_html=True
)


pg.run()