import urllib.request
from bs4 import BeautifulSoup
import nltk
import string

# download stopwords if not already installed
nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.probability import FreqDist


# URL of webpage
url = "https://en.wikipedia.org/wiki/Apple"

# Add header to avoid blocking
headers = {'User-Agent': 'Mozilla/5.0'}

req = urllib.request.Request(url, headers=headers)
response = urllib.request.urlopen(req)

html = response.read()


# Parse HTML
soup = BeautifulSoup(html, "html.parser")

# Extract text
text = soup.get_text()


# Tokenization
tokens = text.split()


# Convert to lowercase
tokens = [word.lower() for word in tokens]


# Remove punctuation
tokens = [word.strip(string.punctuation) for word in tokens]


# Load stopwords
stop_words = set(stopwords.words('english'))


# Remove stopwords
clean_tokens = [word for word in tokens if word not in stop_words and word.isalpha()]


# Frequency distribution
freq = FreqDist(clean_tokens)


# Print word frequencies
for word, count in freq.most_common(50):
    print(word, ":", count)


# Plot top 20 words
freq.plot(20, cumulative=False)