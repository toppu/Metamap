import os

import streamlit as st
from st_pages import Page, Section, add_page_title, show_pages
from streamlit import config as _config

# Force CORS settings for Azure deployment
if os.getenv('WEBSITE_SITE_NAME'):  # Running on Azure
    _config._set_option('server.enableCORS', True, 'command_line')
    _config._set_option('server.enableXsrfProtection', False, 'command_line')
    _config._set_option('server.enableWebsocketCompression', False, 'command_line')


def main():
    show_pages(
        [
            Page("src/pages/Welcome.py", "Toolbox", "🏠", in_section=False),
            Page("src/pages/Instructions.py", "Instructions", " 📝", in_section=False),
            Section(name="Explore", icon="1️⃣"),
            Page("src/pages/Data_Upload.py", "Upload data", "⬆️"),
            Page("src/pages/Statistical_Analysis.py", "Statistical analysis", "📊"),
            Page("src/pages/Ecological_Diversity.py", "Ecological measures", "📖"),
            Page("src/pages/Machine_Learning.py", "AI predictions", "🌐"),
        ]
    )

    add_page_title()

if __name__ == '__main__':

    st.title("Welcome to the microbiome Analysis & AI Prediction toolbox ✨")
    st.write("""
    This website is designed to **facilitate** the data analysis of microbiome datasets. Specifically, the website is designed to compare two groups of microbiome datasets (e.g., case vs control, positive vs negative etc) that are identified through the metadata.
    """)
    st.write("""
    It provides basic exploratory analysis and statistical tests tools. The website also provides various pre-processing tools and Artificial Intelligence (AI) models, including SVM, Logistic Regression, KNN, and XGBoost, to help discover potential patterns and insights in your data.
    """)

    # Display image if available
    import os
    if os.path.exists('./ims/glimpse.png'):
        st.image('./ims/glimpse.png')

    main()