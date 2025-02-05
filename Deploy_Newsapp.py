# IMPORT LIBRARIES
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud, STOPWORDS
import scipy
import nltk
import re
from nltk.stem import PorterStemmer
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from nltk import sent_tokenize
from nltk.corpus import stopwords
from tqdm.notebook import tqdm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from collections import Counter
import torch
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification, BertTokenizer, BertForSequenceClassification
import warnings
import spacy
from spacy import displacy
from umap import UMAP
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
from plotly.graph_objects import Layout
from pprint import pprint
import operator
from gensim import corpora, models
from sklearn.feature_extraction.text import CountVectorizer
import gensim
from gensim.utils import simple_preprocess
from pprint import pprint
import gensim.corpora as corpora
import warnings
pd.options.display.max_columns = None
import itertools
from bertopic import BERTopic
import streamlit as st
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter
#from topicwizard.figures import word_map, document_topic_timeline, topic_wordclouds, word_association_barchart
#from topicwizard.figures import word_map
import streamlit as st
import nltk
import operator
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from bertopic import BERTopic
from umap import UMAP
warnings.filterwarnings("ignore")
import plotly.graph_objects as go
from plotly.graph_objects import Layout
from IPython.display import display
import streamlit as st
from keybert import KeyBERT
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer
import plotly.express as px
import networkx as nx  # Ensure this library is installed
from concurrent.futures import ThreadPoolExecutor

# Upload Data 

data = pd.read_csv('/Users/taichowdhury/Desktop/Finance_NLP/Finance_News_NLP/Data/dataset_nlp.csv', header=None, encoding='latin-1')
data = pd.DataFrame(data)

# Assign Headers
header_names = ['Sentiment', 'Text']  # Replace with your desired header names
# Assign the header names to the DataFrame
data.columns = header_names

# Add additional Features to the Dataframe:

tokens = nltk.word_tokenize(text=str(data['Text']), language='english')

# 'Text' column contains the text data
data['word_counts'] = data['Text'].apply(lambda x: len(word_tokenize(str(x))))
data['Word_Length'] = data['Sentiment'].apply(lambda x: len(str(x)))

# Add SPECIAL CHARACTER COUNTS feature
special_characters = set("$%**#$&'")  # Add your desired special characters to this set
def count_special_characters(text):
    return sum(1 for char in str(text) if char in special_characters)
data['special_char_counts'] = data['Text'].apply(count_special_characters)

# Function for text normalization
def normalize_text(text):
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Tokenize the text
    words = word_tokenize(text)
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word not in stop_words]
    
    # Apply stemming (you can also use lemmatization)
    stemmer = PorterStemmer()
    words = [stemmer.stem(word) for word in words]
    
    # Join the normalized words back into a sentence
    normalized_text = ' '.join(words)
    
    return normalized_text

# Apply text normalization to the 'cleaned_text' column
data['normalized_text'] = data['Text'].apply(normalize_text)

# Download NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

def clean_text(text):
    # Lowercasing
    text = text.lower()
    
    # Removing URLs
    text = re.sub(r'http\S+', '', text)
    text = re.sub('<[^<]+?>', '', text)
    text=re.sub("[^a-zA-Z]+", " ", text)
    
    # Tokenization
    tokens = word_tokenize(text)
    
    # Removing punctuation
    tokens = [word for word in tokens if word.isalnum()]
    
    # Removing stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if not word in stop_words]
    
    # Stemming
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(word) for word in tokens]
    
    # Joining tokens back into text
    clean_text = ' '.join(tokens)
    
    return clean_text

# Example usage
# Assuming 'df' is your DataFrame and 'column_name' is the name of the column containing the text
data['cleaned_text'] = data['normalized_text'].apply(clean_text)

#STOPWORDS 

nltk.download('stopwords')

def remove_stopwords(text):
    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    
    # Get English stopwords
    stop_words = set(stopwords.words('english'))
    
    # Remove stopwords
    filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
    
    # Join the filtered tokens back into a string
    filtered_text = ' '.join(filtered_tokens)
    
    return filtered_text

# Assuming data is a pandas DataFrame
data['cleaned_text'] = data['cleaned_text'].apply(remove_stopwords)



##############################################
# INTRODUCTION #
##############################################

# Set Streamlit to wide mode
st.set_page_config(layout="wide")

# Center the main title
st.markdown("""
    <div style="text-align: center; color: maroon; margin-bottom: 50px;">
        <h1 style="font-size: 3em;">Financial News Topic Modeling Using Natural Language Processing</h1>
    </div>
""", unsafe_allow_html=True)

# Create columns
col1, col2 = st.columns([1, 2])  # Specify the width ratio for the columns

# Add the image in the first column
with col1:
    st.image("/Users/taichowdhury/Desktop/Finance_NLP/Finance_News_NLP/Data/insight.webp", use_column_width=True)

# Add the subtitle and paragraph in the second column with some margin-top
with col2:
    st.markdown("""
        <h2 style='color: maroon;'>BUSINESS UNDERSTANDING</h2>
    """, unsafe_allow_html=True)
    st.write("""
        <div style="text-indent: 40px; margin-top: 20px;">
            One of the fund managers at JP Morgan has enlisted our support to derive valuable insights from financial news analysis. We have a dataset featuring sentiments (categorized as positive, negative, or neutral) and text snippets from financial news articles. Using AI models, our objectives are to identify and analyze crucial economic indicators within the text, including GDP, Inflation, Interest Rates, and Employment; categorize news articles into topics to help managers understand the impact of factors such as financial markets, geopolitical events, and trade agreements on specific industries like energy stocks; and extract insights on market dynamics to keep managers informed about various financial news developments. This comprehensive analysis will aid the fund managers in making informed decisions by understanding the broader implications of financial news for their portfolios.
        </div>
    """, unsafe_allow_html=True)


###################################
# DATE PREPROCESSING             #
##################################

###############################
# STREAMLIT FIGURE 1: WORDCLOUD
###############################

# Inject custom CSS to make the display wider
st.markdown(
    """
    <style>
    .main {
        max-width: 32000px; /* Increase this value as needed */
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Combine text data
combined_text = ' '.join(data['cleaned_text'].tolist())

# Calculate word frequencies
word_frequencies = Counter(combined_text.split())
top_four_words = [word for word, freq in word_frequencies.most_common(4)]

# Define a function to generate font color for the WordCloud
def custom_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
    if word in top_four_words:
        return "#800000"  # Maroon color for top 4 words
    else:
        return "grey"  # Grey color for other words

# Create WordCloud object with custom color function
wordcloud2 = WordCloud(background_color='#f8f8f8', 
                       width=1200, height=800,
                       color_func=custom_color_func,  # Use custom color function
                       max_font_size=110, min_font_size=5,
                       collocation_threshold=2).generate(combined_text)

# Define function to generate word cloud for specific sentiment category
def wordcount_gen(df, category):
    combined_tweets = " ".join(tweet for tweet in df[df.Sentiment == category]['cleaned_text'])
    
    def maroon_color_func(word, font_size, position, orientation, random_state=None, **kwargs):
        return "#800000"  # Hex code for maroon

    wc = WordCloud(background_color='white', 
                   max_words=50, 
                   stopwords=STOPWORDS,
                   color_func=maroon_color_func)  # Apply the maroon color function

    wordcloud = wc.generate(combined_tweets)
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Word Cloud Generator for Financial News</h1>", unsafe_allow_html=True)

# Create columns for the layout
main_col, right_col = st.columns([2, 1])

# Main word cloud on the left
with main_col:
    st.markdown('<h1 style="text-align: center; color: #370617; font-family: Traditionalist;">Word Cloud</h1>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.imshow(wordcloud2, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)

# Sidebar toolbar for buttons
with st.sidebar:
    st.markdown('<h2 style="color: #370617; font-family: Traditionalist;">Generate Wordcloud By Sentiment</h2>', unsafe_allow_html=True)
    
    positive_clicked = st.button("Generate Positive Word Cloud")
    negative_clicked = st.button("Generate Negative Word Cloud")
    neutral_clicked = st.button("Generate Neutral Word Cloud")

# Summary and sentiment word clouds on the right
with right_col:
    st.markdown('<h2 style="text-align: center; color: #370617; font-family: Traditionalist;">Summary</h2>', unsafe_allow_html=True)

    summary = "A word cloud shows word frequency in a text. Our word cloud analysis shows 'EU MN,' 'EUR MILLION,' 'OPER PROFIT,' and 'NET SALE' are common. We also found positive ('EUR,' 'MN,' 'FINNISH,' 'COMPANI'), neutral ('SERVICE,' 'COMPANI,' 'FINNISH,' 'OPER'), and negative ('FINNISH,' 'COMPANI,' 'EUR,' 'MN') sentiments. This suggests the dataset is about European Union finance, with mixed sentiment about companies and operations."

    st.write(summary)
    
    st.markdown('<h2 style="text-align: center; color: #370617; font-family: Traditionalist;">Sentiment Word Cloud</h2>', unsafe_allow_html=True)
    
    if positive_clicked:
        wordcount_gen(data, 'Positive')
    elif negative_clicked:
        wordcount_gen(data, 'Negative')
    elif neutral_clicked:
        wordcount_gen(data, 'Neutral')

########################################################
##### N GRAM CHARTS ##############
########################################################


nltk.download('punkt')

def compute_ngrams(tok, n):
    """
    Function to compute n-grams like bigram, trigram etc.
    tok: tokenized words
    n: Any integer to get the number of words together. n=2 bigram, n=3 trigram
    """
    return list(zip(*(tok[index:] for index in range(n))))

def top_ngram(text_corpus, ngram_val=4, limit=5):
    """
    Function that helps us get the top n-gram from a bunch of text.
    text_corpus: pass in clean text (strings)
    ngram_val: Pass in integers 2 for bigram, 3 for trigram, etc.
    limit: Show the top number of top bigram, trigram, etc. (5,10,15, etc.)
    """
    token = word_tokenize(text_corpus)
    ngram = compute_ngrams(token, ngram_val)
    ngram_freq_dist = nltk.FreqDist(ngram)
    sorted_ngrams_fd = sorted(ngram_freq_dist.items(), key=operator.itemgetter(1), reverse=True)
    sorted_ngrams = sorted_ngrams_fd[:limit]
    sorted_ngrams = [(' '.join(text), freq) for text, freq in sorted_ngrams]
    
    grams = [text for text, freq in sorted_ngrams]
    grams_val = [freq for text, freq in sorted_ngrams]
    
    plt.figure(figsize=(6, 4))  # Adjust size as needed
    sns.barplot(x=grams_val, y=grams, color='#800000')  # Setting the bars to maroon color
    plt.title(f'Top most used {ngram_val}-grams', fontsize=20, fontweight='bold')
    plt.xlabel('Frequency', fontsize=15)
    plt.ylabel('N-Gram', fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    st.pyplot(plt)
    
    return sorted_ngrams

# Initialize the summarization pipeline
summarizer = pipeline("summarization")

def get_summary(text, max_length):
    try:
        summary = summarizer(text, max_length=max_length, min_length=30, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return str(e)

# Streamlit UI
#st.markdown("<h1 style='text-align: center;'>Text Analysis Tool</h1>", unsafe_allow_html=True)




    
        
############################################

############################################

# Streamlit UI
st.markdown("<h1 style='text-align: center;'>Text Analysis Dashboard</h1>", unsafe_allow_html=True)

# Optionally display the DataFrame
if st.checkbox("Show DataFrame"):
    st.write(data)

# Create two columns
col1, col2 = st.columns(2)

# Generate histogram in the first column
with col1:
    st.subheader("Histogram of Text Lengths")
    fig1, ax1 = plt.subplots()
    data['cleaned_text'].str.len().hist(ax=ax1, bins=20, color='#800000')  # Setting the bars to maroon color
    ax1.set_title('Distribution of Text Lengths')
    ax1.set_xlabel('Text Length')
    ax1.set_ylabel('Frequency')
    st.pyplot(fig1)

# Generate sentiment count plot in the second column
with col2:
    st.subheader("Sentiment Count Plot")
    y = data['Sentiment']
    fig2, ax2 = plt.subplots(figsize=(6, 4))
    sns.countplot(y, ax=ax2, color='#800000')
    ax2.set_xlabel('Count')  # Label for the X-axis
    ax2.set_ylabel('')  # Label for the Y-axis
    ax2.set_title('Sentiment Count')  # Add a title to the plot
    st.pyplot(fig2)


##################################################
####### VISUALIZE PROB CHART ###################
#################################################
import streamlit as st
import plotly.graph_objects as go
from plotly.graph_objects import Layout
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd  # Assuming you are using pandas for the series

# Define the first visualization function
def create_probability_graph_plotly(digits, probabilities, text_):
    custom_labels = {
        -1: 'Business Service',
         0: 'Operating Profit in millions Eur',
         1: 'Australian Mineral Companies',
         2: 'Nordic Exchange - Helsinki',
         3: 'Newspaper/Publish Media',
         4: 'Cyber Data Intelligence',
         5: 'Beer/Breweries',
         6: 'Energy Power Plant',
         7: 'St petersburg',
         8: 'Sales Expectation',
         9: 'Revenue Growth',
        10: 'Abroad Sales in Finland',
        11: 'Share Capital',
        12: 'Stake fortum',
        13: 'Value Contract',
        14: 'Dividend Per Share',
        15: 'Million Euros of Finnish Metal',
        16: 'Accord Case Rule',
        17: 'Pre-tax profit',
        18: 'Electron Manufacture'
    }
    
    # Generate labels for cluster topics
    for digit in set(digits):
        if digit not in custom_labels:
            custom_labels[digit] = f'Cluster {digit}'
    
    # Map the numeric digits to custom labels
    x_axis_labels = [custom_labels[digit] for digit in digits]
    
    # Define colors for bars based on probability
    bar_colors = ['#adb5bd' if prob <= 0.7 else 'maroon' for prob in probabilities]
    
    # Create bar graph using plotly
    fig = go.Figure()
    fig.add_trace(go.Bar(x=probabilities, y=x_axis_labels, marker_color=bar_colors, orientation='h'))

    # Add annotations for each bar
    for prob, label in zip(probabilities, x_axis_labels):
        fig.add_annotation(
            text=f"{prob:.2f}",  # Display probability value with two decimal places
            x=prob,  # x-coordinate position of the annotation (probability value)
            y=label,  # y-coordinate position of the annotation (topic label)
            showarrow=False,  # Disable arrow
            font=dict(color='black', size=12),  # Set font color to black
            xshift=20  # Adjust horizontal position of annotation
        )

    # Set labels and layout
    fig.update_layout(
        yaxis_title='<b>Topics</b>',
        xaxis=dict(tickfont=dict(size=12, color='black'), showgrid=False, showline=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=12, color='black'), autorange="reversed"),  # Change y-axis tick font color to black
        plot_bgcolor='rgba(0,0,0,0)', 
        paper_bgcolor='rgba(0,0,0,0)', 
        font=dict(family="Arial", size=12, color='black'),  # Set overall font color to black
        width=800, 
        height=600
    )
    
    return fig

# Define the second visualization function
def create_frequency_barplot(topic_counts):
    custom_labels = {
        -1: 'Business Service',
        0: 'Operating Profit in millions Eur',
        1: 'Australian Mineral Companies',
        2: 'Nordic Exchange - Helsinki',
        3: 'Newspaper/Publish Media',
        4: 'Cyber Data Intelligence',
        5: 'Beer/Breweries',
        6: 'Energy Power Plant',
        7: 'St Petersburg',
        8: 'Sales Expectation',
        9: 'Revenue Growth',
        10: 'Abroad Sales in Finland',
        11: 'Share Capital',
        12: 'Stake fortum',
        13: 'Value Contract',
        14: 'Dividend Per Share',
        15: 'Million Euros of Finnish Metal',
        16: 'Accord Case Rule',
        17: 'Pre-tax profit',
        18: 'Electron Manufacture'
    }

    # Sort the topic counts in descending order
    topic_counts_sorted = topic_counts.sort_values(ascending=False)

    # Map topic indices to their respective names
    topic_names_ = pd.DataFrame({
        'Name': [custom_labels[idx] for idx in topic_counts_sorted.index],
        'Count': topic_counts_sorted.values
    })

    total = np.sum(topic_names_['Count'])

    # Determine bar colors based on the 10% threshold
    colr = ['maroon' if (x / total) > 0.1199 else '#adb5bd' for x in topic_names_['Count']]

    fig, ax = plt.subplots(figsize=(5, 2.5), facecolor='none')  # Further reduce the figure size

    sns.barplot(data=topic_names_, x='Count', y='Name', palette=colr, ax=ax)

    for p in ax.patches:
        x = p.get_x() + p.get_width()
        y = p.get_y() + p.get_height() / 2
        barThickness = p.get_width()
        label = '{:.1f}%'.format(100 * barThickness / total)
        ax.annotate(
            label,
            (x, y),
            ha='center',
            va='center',
            size=6,  # Further reduce annotation font size
            color='black',  # Set annotation color to black
            xytext=(30, 1),  # Offset the annotation text by 30 points to the right and 1 point up
            textcoords='offset points'
        )

    ax.set_ylabel('Theme Words', fontweight='bold', size=6, color='black')  # Further reduce label font size
    ax.set_xlabel('')  # Remove x-axis label
    ax.set_yticklabels(topic_names_['Name'], fontsize=6, color='black')  # Further reduce tick label font size

    # Remove ticks from x-axis
    ax.xaxis.set_ticks([])
    ax.xaxis.set_ticklabels([])

    # Disable x-axis line
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)  # Hide the top border
    ax.spines['right'].set_visible(False)  # Hide the right border
    ax.spines['left'].set_visible(False)  # Hide the left border

    # Make the background transparent
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)

    plt.tight_layout()
    return fig

# Streamlit app
st.markdown("<h1 style='text-align: center;'>NLP Topic Model Visualization</h1>", unsafe_allow_html=True)

# Placeholder data for demonstration
test = "sample text"
topic_counts = pd.Series([100, 200, 150, 300, 250, 180, 220, 270, 260, 240], index=range(10))

digits = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
probabilities = [0.95, 0.85, 0.78, 0.67, 0.60, 0.59, 0.57, 0.55, 0.53, 0.50]
text_ = "Sample text for topic model"

# Create two columns
col1, col2 = st.columns([1.2, 1.8])  # Adjust column width ratios

# Summarized paragraph in the first column
with col1:
    st.markdown("<br><br><br><br><br><br><br><br>", unsafe_allow_html=True)  # Add spacing to push the paragraph down
    st.markdown("""
    This chart shows the probability distribution for topics. The y-axis lists the topics identified by the model, while the x-axis shows their probabilities.

    High probabilities for topics like "Nordic Exchange - Helsinki" and "Australian Mineral Companies" indicate their prominence in the dataset. A 95% probability for "Operating Profit in Millions EUR" highlights the text's strong relevance to this topic.

    Topics with probabilities above 60% are marked in maroon, emphasizing their importance. Dominant themes include 'News/Publish Media,' 'Nordic Exchange - Helsinki,' 'Australian Mineral Companies,' and 'Operating Profit in Millions EUR.'
    """, unsafe_allow_html=True)

# Chart and title in the second column
with col2:
    st.markdown("<h4 style='text-align: center; margin-bottom: -20px;'>Probability Distribution for Topics</h4>", unsafe_allow_html=True)
    fig1 = create_probability_graph_plotly(digits, probabilities, text_)
    st.plotly_chart(fig1)

# Create two columns for the second visual
col3, col4 = st.columns([1, 1.5])  # Adjust column width

# Add new paragraph in the first column of the new row
with col3:
    st.markdown("<br><br>", unsafe_allow_html=True)  # Reduce space above the paragraph
    st.markdown("""
    The "Top 10 Theme Phrases" bar plot shows the most frequent topics in the dataset based on their occurrence. Topics like 'Newspaper/Publish Media', 'St Petersburg', 'Sales Expectation', and 'Cyber Data Intelligence' have the highest counts, with 'Newspaper/Publish Media' being the most prominent at 13.8%. This indicates its significant presence in the corpus.
    """, unsafe_allow_html=True)

# Display second visual in the second column of the new row
with col4:
    st.markdown("<h4 style='text-align: center;'>Most Frequent Top 10 Theme Phrases</h4>", unsafe_allow_html=True)
    fig2 = create_frequency_barplot(topic_counts)
    st.pyplot(fig2)


##########################
# STREAMLIT FIGURE 3: BERT TOPIC CLUSTERING
##########################



# Check if DataFrame has the necessary columns
if 'Text' not in data.columns:
    st.error("DataFrame must contain a 'cleaned_text' column.")
else:
    # Title: Comprehensive Topic Model Analysis
    st.markdown("""
        <div style="text-align: center; color: maroon; margin-bottom: 50px;">
            <h1 style="font-size: 3em;">Comprehensive Topic Model Analysis</h1>
        </div>
    """, unsafe_allow_html=True)

    # BERT Topic Clustering Heading
    st.markdown("<h2 style='text-align: center; color: maroon;'>BERT Topic Clustering</h2>", unsafe_allow_html=True)

    # Initialize sentence transformer and get embeddings
    from sentence_transformers import SentenceTransformer
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = sentence_model.encode(data['Text'].tolist(), show_progress_bar=True)

    # Perform UMAP dimensionality reduction
    from umap import UMAP
    umap_model = UMAP(n_neighbors=15, n_components=2, metric='cosine')
    umap_embeddings = umap_model.fit_transform(embeddings)

    # Perform KMeans clustering
    from sklearn.cluster import KMeans
    kmeans_model = KMeans(n_clusters=5, random_state=42)
    clusters = kmeans_model.fit_predict(embeddings)
    data['cluster'] = clusters

    # Perform BERTopic topic modeling
    from bertopic import BERTopic
    bertopic_model = BERTopic(umap_model=umap_model, verbose=True)
    topics, _ = bertopic_model.fit_transform(data['Text'].tolist())
    data['topic'] = topics

    # Custom labels for topics
    custom_labels = {
        -1: 'Business Service',
        0: 'Operating Profit in millions Eur',
        1: 'Australian Mineral Companies',
        2: 'Nordic Exchange - Helsinki',
        3: 'Newspaper/Publish Media',
        4: 'Cyber Data Intelligence',
        5: 'Beer/Breweries',
        6: 'Energy Power Plant',
        7: 'St Petersburg',
        8: 'Sales Expectation',
        9: 'Revenue Growth',
        10: 'Abroad Sales in Finland',
        11: 'Share Capital',
        12: 'Stake fortum',
        13: 'Value Contract',
        14: 'Dividend Per Share',
        15: 'Million Euros of Finnish Metal',
        16: 'Accord Case Rule',
        17: 'Pre-tax profit',
        18: 'Electron Manufacture'
    }
    data['topic_label'] = data['topic'].map(custom_labels)

    # Create columns for the intertopic map and the summary paragraph
    col1, col2 = st.columns([2, 1])  # Adjust column widths as needed

    # Intertopic Distance Map in the first column
    with col1:
        fig = bertopic_model.visualize_topics()
        st.markdown("<h3 style='text-align: center; color: maroon;'>Intertopic Distance Map</h3>", unsafe_allow_html=True)
        st.plotly_chart(fig, use_container_width=True)

    # Paragraph in the second column
    with col2:
        st.markdown("<h3 style='text-align: center; color: maroon;'>Analysis Summary</h3>", unsafe_allow_html=True)
        st.write("""
        An intertopic distance map is a visualization technique used to understand the relationships between different topics discovered by a topic modeling algorithm. In this map, each topic is represented as a point, and the distance between points indicates the similarity or dissimilarity between the corresponding topics.

In your case, the fact that most of the 96 topics detected by your BERTopic model fall into quadrants 1 and 3 suggests a certain pattern in the relationships between these topics. Topics in quadrant 1 are typically considered to be more similar to each other than topics in other quadrants, while topics in quadrant 3 might represent more distinct or contrasting themes.

The identification of three best topic clusters - service/construction, biotech/glass, and financials/operations - provides valuable insights into the key themes present in your corpus. These clusters likely represent distinct areas of discussion or focus within your dataset.
        """)

    
    import concurrent.futures  # For parallel processing
    

    # Initialize the DistilBERT zero-shot classification pipeline with caching
    @st.cache_resource
    def load_classifier():
        return pipeline("zero-shot-classification", model="valhalla/distilbart-mnli-12-1")

    classifier = load_classifier()

    # Define candidate labels (themes)
    candidate_labels = [
        'Operating Profit in Millions EUR', 'Australian Mineral Companies',
        'Nordic Exchange - Helsinki', 'Newspaper/Publish Media',
        'Cyber Data Intelligence', 'Beer/Breweries', 'Energy Power Plant',
        'St Petersburg', 'Sales Expectation', 'Revenue Growth',
        'Abroad Sales in Finland', 'Share Capital', 'Stake Fortum',
        'Value Contract', 'Dividend Per Share', 'Million Euros of Finnish Metal',
        'Accord Case Rule', 'Pre-tax Profit', 'Electron Manufacture'
    ]

    # Function to detect theme for a single text
    def detect_theme(text):
        try:
            result = classifier(text, candidate_labels)
            return result['labels'][0]  # Get the top theme
        except Exception as e:
            return "Error"

    # Process text in parallel using ThreadPoolExecutor for speed
    def process_themes_in_parallel(texts):
        with concurrent.futures.ThreadPoolExecutor() as executor:
            themes = list(executor.map(detect_theme, texts))
        return themes
    
    # Generate a pie chart of the top 10 themes
    def plot_top_themes_pie(data):
        theme_counts = data['Assigned_Theme_Name'].value_counts().nlargest(10).reset_index()
        theme_counts.columns = ['Theme', 'Count']

        # Create an interactive pie chart using Plotly
        fig = px.pie(
            theme_counts, 
            names='Theme', 
            values='Count', 
            title='Top 10 Themes Distribution',
            hole=0.3
        )
        fig.update_traces(textposition='inside', textinfo='percent+label')
        fig.update_layout(showlegend=True, title_x=0.5)

        # Display the pie chart
        st.plotly_chart(fig)

    # Limit the data to top 500 rows to speed up processing (optional)
    sample_data = data.head(500)

    # Perform theme detection in parallel
    sample_data['Assigned_Theme_Name'] = process_themes_in_parallel(sample_data['cleaned_text'])

    # Load zero-shot classification model with caching
    @st.cache_resource
    def load_classifier():
        return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Initialize classifier and define candidate labels
    classifier = load_classifier()
    candidate_labels = [
        'Operating Profit in Millions EUR', 'Australian Mineral Companies', 
        'Nordic Exchange - Helsinki', 'Newspaper/Publish Media',
        'Cyber Data Intelligence', 'Beer/Breweries', 
        'Energy Power Plant', 'St Petersburg', 'Sales Expectation',
        'Revenue Growth', 'Abroad Sales in Finland', 'Share Capital',
        'Stake Fortum', 'Value Contract', 'Dividend Per Share',
        'Million Euros of Finnish Metal', 'Accord Case Rule', 
        'Pre-tax Profit', 'Electron Manufacture'
    ]

    # Function to detect theme for a single text
    def detect_theme(text):
        result = classifier(text, candidate_labels)
        return result['labels'][0]  # Get the top theme

    # Process themes in parallel
    def process_themes_in_parallel(texts):
        with ThreadPoolExecutor() as executor:
            themes = list(executor.map(detect_theme, texts))
        return themes

    # Sample data for faster processing
    sample_data = data.sample(n=min(500, len(data)), random_state=42)  # Limit data to 500 rows

    # Apply theme detection in parallel
    sample_data['Assigned_Theme_Name'] = process_themes_in_parallel(sample_data['cleaned_text'])

    plot_top_themes_pie(sample_data)


# Display the pie chart

    # Sentiment Analysis Using NLTK
    #st.markdown("<h2 style='color: maroon;'>Sentiment Analysis Using NLTK</h2>", unsafe_allow_html=True)

    sia = SentimentIntensityAnalyzer()

    # Function to format NLTK sentiment scores
    def format_nltk_sentiment_output(sentiment_dict):
        neg = f"Neg {sentiment_dict['neg']:.3f}"
        neu = f"Neu {sentiment_dict['neu']:.3f}"
        pos = f"Pos {sentiment_dict['pos']:.3f}"
        return f"{neg}, {neu}, {pos}"

    # Get sentiment scores and apply formatting
    data['sentiment_scores'] = data['Text'].apply(lambda x: format_nltk_sentiment_output(sia.polarity_scores(x)))

    # Extract compound score for analysis (optional)
    data['compound'] = data['Text'].apply(lambda x: sia.polarity_scores(x)['compound'])


    # Increase the maximum column width for display
    #pd.set_option('display.max_colwidth', None)
    
    # Display the data with full text
    #st.dataframe(data[['Text', 'sentiment_scores']].head(10))

    # BERT-Based Sentiment Analysis
    st.markdown("<h2 style='text-align: center; color: maroon;'>BERT Based Sentiment Analysis</h2>", unsafe_allow_html=True)

    # Initialize the sentiment analysis pipeline
    sentiment_analysis = pipeline(
        "sentiment-analysis", 
        model="distilbert-base-uncased-finetuned-sst-2-english", 
        framework="pt"
    )

    # Function to format BERT sentiment output
    def format_bert_sentiment_output(result):
        label = result['label'].capitalize()  # Convert to "Positive", "Negative", or "Neutral"
        score = round(result['score'], 2)     # Round the score to 2 decimal places
        return f"{label} ({score:.2f})"       # Example: "Positive (0.85)"

    # Apply the sentiment analysis and formatting
    data['bert_sentiment'] = data['Text'].apply(
        lambda x: format_bert_sentiment_output(sentiment_analysis(x)[0])
    )

    # Display the DataFrame with formatted results  
    st.dataframe(data[['Text', 'bert_sentiment']].head(10))

    # Create two columns in the Streamlit app
    col1, col2 = st.columns(2)

    with col1:
        # Sentiment Score Histogram
        st.markdown("<h2 style='color: maroon;'>Sentiment Score Histogram</h2>", unsafe_allow_html=True)

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.histplot(data['compound'], kde=True, color='skyblue', ax=ax)
        ax.set_title('Distribution of Sentiment Scores', fontsize=18, pad=20)
        ax.set_xlabel('Sentiment Score')
        ax.set_ylabel('Frequency')
        sns.despine()
        st.pyplot(fig)

    with col2:
        # Sentiment Scores by Topic
        st.markdown("<h2 style='color: maroon;'>Sentiment Scores by Topic</h2>", unsafe_allow_html=True)

        if 'topic_label' in data.columns and 'compound' in data.columns:
            if not data[['topic_label', 'compound']].dropna().empty:
                # Filter to only the top 10 topics
                top_topics = data['topic_label'].value_counts().nlargest(10).index
                filtered_data = data[data['topic_label'].isin(top_topics)]

                fig, ax = plt.subplots(figsize=(12, 8))
                sns.boxplot(data=filtered_data, x='topic_label', y='compound', palette='coolwarm', ax=ax)
                ax.set_title('Sentiment Scores by Top 10 Topics', fontsize=18, pad=20)
                ax.set_xlabel('Topic')
                ax.set_ylabel('Sentiment Score')
                ax.tick_params(axis='x', rotation=45)
                
                sns.despine()  # Remove all borders and gridlines
                st.pyplot(fig)
            else:
                st.write("**The 'topic_label' or 'compound' columns contain missing values.**")
        else:
            st.write("**'topic_label' or 'compound' columns are missing or empty.**")


import streamlit as st
import pandas as pd
import plotly.express as px
from transformers import pipeline

@st.cache_resource
def load_sentiment_model():
    return pipeline('sentiment-analysis', model='nlptown/bert-base-multilingual-uncased-sentiment')

@st.cache_resource
def load_theme_model():
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

# Sidebar for parameters
with st.sidebar:
    st.header("Parameters For Analysis Tool")

    # Slider for selecting n-gram value
    ngram_val = st.slider("Select N-Gram Value", min_value=2, max_value=5, value=4)

    # Slider for selecting the limit
    limit = st.slider("Select Number of Top N-Grams to Display", min_value=1, max_value=20, value=5)

    # Text summarization options
    summary_length = st.selectbox("Select Summary Length", ["20 words", "30 words", "50 words"])

# Input for text corpus (prompt)
text_corpus = st.text_area("Enter Text Corpus", placeholder="Type or paste the text you want to analyze here")

# Button for generating summary, Probability Distribution, and sentiment analysis
generate_ngram_summary = st.button("Generate Summary, Probability Distribution, and Sentiment Analysis")

# Main area for displaying results
if generate_ngram_summary:
    if text_corpus.strip():  # Check if text corpus is not empty
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Text Summary")
            max_length = int(summary_length.split()[0])
            summary = get_summary(text_corpus, max_length)
            st.write(summary)

        # ---- Probability Distribution by Topic ----
        st.subheader("Probability Distribution by Topic")

        # Load the cached theme model
        theme_model = load_theme_model()
        candidate_labels = ["Politics", "Technology", "Finance", "Health", "Education", "Entertainment"]

        # Get theme prediction probabilities
        theme_result = theme_model(text_corpus, candidate_labels)
        labels = theme_result['labels']
        scores = [round(score, 2) for score in theme_result['scores']]  # Round probabilities

        # Create a DataFrame for the results
        theme_df = pd.DataFrame({'Topic': labels, 'Probability': scores})

        # Identify the color: Maroon for the highest, grey for the rest
        colors = ['maroon' if prob == max(scores) else 'grey' for prob in scores]

        # Create a bar plot using Plotly
        fig = px.bar(theme_df, x='Probability', y='Topic', orientation='h',
                     title="Topic Probability Distribution",
                     labels={'Topic': 'Topics', 'Probability': 'Probability'},
                     text='Probability')

        # Update bar colors and layout
        fig.update_traces(marker_color=colors, texttemplate='%{x:.2f}')
        fig.update_layout(yaxis_categoryorder='total ascending', title_x=0.5)

        st.plotly_chart(fig)  # Display the bar plot

        # ---- NetworkX Visualization ----
        st.subheader("NetworkX Graph Visualization")

        # Create a simple NetworkX graph
        G = nx.Graph()
        for i, topic in enumerate(labels):
            G.add_node(topic, size=scores[i])  # Add nodes with sizes based on probability
            if i > 0:
                G.add_edge(labels[i - 1], topic)  # Connect nodes sequentially

        # Draw the graph using Matplotlib
        fig, ax = plt.subplots(figsize=(8, 5))
        pos = nx.spring_layout(G, seed=42)  # Positioning the nodes
        nx.draw(G, pos, with_labels=True, node_size=[v * 1000 for v in scores], node_color='skyblue', ax=ax)
        st.pyplot(fig)  # Display the graph in Streamlit

        # ---- BERT Sentiment Analysis ----
        st.subheader("BERT Sentiment Analysis")

        # Load the cached sentiment analysis pipeline
        sentiment_analysis = load_sentiment_model()

        # Perform sentiment analysis on the text corpus
        sentiment_result = sentiment_analysis(text_corpus)[0]

        st.write(f"Sentiment: {sentiment_result['label']}, Score: {sentiment_result['score']:.2f}")

    else:
        st.error("Please enter a valid text corpus.")


from transformers import pipeline
import pandas as pd
import streamlit as st
import plotly.express as px


# Streamlit File Upload for Corpus Analysis
uploaded_file = st.file_uploader("Upload Trained Corpus File", type=["csv", "txt"])

if uploaded_file:
    try:
        # Try reading the file with UTF-8 encoding
        uploaded_data = pd.read_csv(uploaded_file, header=None, encoding='utf-8')
    except UnicodeDecodeError:
        # If UTF-8 fails, try with Latin-1 encoding
        uploaded_file.seek(0)  # Reset file pointer
        uploaded_data = pd.read_csv(uploaded_file, header=None, encoding='latin-1')
    except pd.errors.EmptyDataError:
        st.error("The uploaded file is empty or incorrectly formatted.")
        st.stop()

    # Ensure the DataFrame has at least one row of data
    if uploaded_data.empty:
        st.error("The uploaded file is empty. Please upload a valid file.")
        st.stop()

    # Rename the single column to 'Text'
    if uploaded_data.shape[1] == 1:
        uploaded_data.columns = ['Text']

    # Clean special characters (optional)
    uploaded_data['Text'] = uploaded_data['Text'].str.encode('ascii', 'ignore').str.decode('ascii')

    # Display the uploaded text data
    st.success("File uploaded successfully! Proceeding with analysis...")
    st.dataframe(uploaded_data.head(10))

    # Initialize the zero-shot classification pipeline
    classifier = pipeline("zero-shot-classification", 
                          model="facebook/bart-large-mnli")

    # Define the candidate labels (themes)
    candidate_labels = [
        'Operating Profit in Millions EUR', 'Australian Mineral Companies', 
        'Nordic Exchange - Helsinki', 'Newspaper/Publish Media',
        'Cyber Data Intelligence', 'Beer/Breweries', 
        'Energy Power Plant', 'St Petersburg', 'Sales Expectation',
        'Revenue Growth', 'Abroad Sales in Finland', 'Share Capital',
        'Stake Fortum', 'Value Contract', 'Dividend Per Share',
        'Million Euros of Finnish Metal', 'Accord Case Rule', 
        'Pre-tax Profit', 'Electron Manufacture'
    ]

    # Function to classify and assign themes using the zero-shot model
    def assign_theme(text):
        result = classifier(text, candidate_labels)
        # Get the label with the highest score
        best_label = result['labels'][0]
        return best_label

    # Apply the function to assign themes
    uploaded_data['Assigned_Theme_Name'] = uploaded_data['Text'].apply(assign_theme)

    # Display the DataFrame with assigned themes
    #st.markdown("<h2 style='color: maroon;'>Assigned Themes</h2>", unsafe_allow_html=True)
    #st.dataframe(uploaded_data[['Text', 'Assigned_Theme_Name']].head(10))

    # KeyBERT keyword extraction (optional)
    from keybert import KeyBERT
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    keybert_model = KeyBERT(model=sentence_model)

    keywords = keybert_model.extract_keywords(
        ' '.join(uploaded_data['Text'].tolist()), 
        keyphrase_ngram_range=(1, 2), 
        stop_words='english', 
        top_n=5
    )

    # Display the extracted keywords
    st.subheader("KeyBERT Keyword Extraction")
    keywords_df = pd.DataFrame(keywords, columns=["Keyword", "Score"])
    fig = px.bar(keywords_df, x='Score', y='Keyword', orientation='h',
                 title="Top Keywords by KeyBERT", labels={'Keyword': 'Keywords', 'Score': 'Relevance Score'})
    fig.update_layout(yaxis_categoryorder='total ascending', title_x=0.5)
    st.plotly_chart(fig)

    # BERT sentiment analysis with improved formatting
    sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

    def format_sentiment_output(result):
        label = result['label'].capitalize()
        score = round(result['score'], 2)
        return f"{label} ({score})"

    uploaded_data['BERT_Sentiment'] = uploaded_data['Text'].apply(
        lambda x: format_sentiment_output(sentiment_analysis(x)[0])
    )

    # Display sentiment analysis results
    st.markdown("<h2 style='color: maroon;'>BERT Sentiment Analysis</h2>", unsafe_allow_html=True)
    st.dataframe(uploaded_data[['Text', 'BERT_Sentiment', 'Assigned_Theme_Name']].head(10))















