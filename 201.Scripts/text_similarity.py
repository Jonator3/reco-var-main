import nltk
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import gst_calculation
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel

nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')
nltk.download('punkt')


def length_difference(text1, text2):
    diff = abs(len(text1) - len(text2))
    return diff, diff/max(len(text1), len(text2)), []


def levenshtein_distance(token1, token2):
    distances = np.zeros((len(token1) + 1, len(token2) + 1))

    for t1 in range(len(token1) + 1):
        distances[t1][0] = t1

    for t2 in range(len(token2) + 1):
        distances[0][t2] = t2

    a = 0
    b = 0
    c = 0

    for t1 in range(1, len(token1) + 1):
        for t2 in range(1, len(token2) + 1):
            if (token1[t1 - 1] == token2[t2 - 1]):
                distances[t1][t2] = distances[t1 - 1][t2 - 1]
            else:
                a = distances[t1][t2 - 1]
                b = distances[t1 - 1][t2]
                c = distances[t1 - 1][t2 - 1]

                if (a <= b and a <= c):
                    distances[t1][t2] = a + 1
                elif (b <= a and b <= c):
                    distances[t1][t2] = b + 1
                else:
                    distances[t1][t2] = c + 1

    out = int(distances[len(token1)][len(token2)])
    return out, out/max(len(token1), len(token2)), []


def token_levenshtein_distance(text1, text2, lemmatize=False):
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return levenshtein_distance(tokens1, tokens2)


def __equal_till(s1, s2):
    for i in range(min(len(s1), len(s2))):
        if s1[i] != s2[i]:
            return i
    return min(len(s1), len(s2))


def longest_common_substring(text1, text2):
    lcs = 0
    start_t1 = 0
    start_t2 = 0
    for i1 in range(len(text1)):
        for i2 in range(len(text2)):
            l = __equal_till(text1[i1:], text2[i2:])
            if l > lcs:
                lcs = l
                start_t1 = i1
                start_t2 = i2
    return lcs, lcs/max(len(text1), len(text2)), [(0, start_t1, start_t1+lcs, 0), (1, start_t2, start_t2+lcs, 0)]


def longest_common_tokensubstring(text1, text2, lemmatize=False):
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return longest_common_substring(tokens1, tokens2)


def gst(string1, string2):
    res = gst_calculation.gst.calculate(string1, string2)
    length = res[-1]
    marking = []
    for section in res[0]:
        t1 = section.get("token_1_position")
        t2 = section.get("token_2_position")
        marking.append((0, t1, t1+section.get("length"), 0))
        marking.append((1, t2, t2+section.get("length"), 0))

    return length, length/max(len(string1), len(string2)), marking


def token_gst(text1, text2, lemmatize=False):
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    return gst(tokens1, tokens2)


def vector_cosine(text1, text2, lemmatize=False):
    if text1 == text2:
        return 1, 1, []
    # Tokenize and lemmatize the texts
    tokens1 = word_tokenize(text1)
    tokens2 = word_tokenize(text2)
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens1 = [lemmatizer.lemmatize(token) for token in tokens1]
        tokens2 = [lemmatizer.lemmatize(token) for token in tokens2]

    if len(tokens1) == 0 or len(tokens2) == 0:
        return 0, 0, []

    # Remove stopwords
    stop_words = stopwords.words('english')
    tokens1 = [token for token in tokens1 if token not in stop_words]
    tokens2 = [token for token in tokens2 if token not in stop_words]

    # Create the TF-IDF vectors
    vectorizer = TfidfVectorizer()
    try:
        vectorizer.fit(tokens1+tokens2)
    except ValueError:
        return 1, 1, []

    try :
        vector1 = vectorizer.transform(tokens1)
        vector2 = vectorizer.transform(tokens2)
    except ValueError:
        print("ValueError in Vcos:")
        print(tokens1)
        print(tokens2)
        print()
        return 0, 0, []
    vector1 = np.asarray(vector1.sum(axis=0)[0])
    vector2 = np.asarray(vector2.sum(axis=0)[0])

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    return similarity[0][0], similarity[0][0], []


def vectorize_with_bert(text):
    # Load the BERT model and tokenizer
    model_name = 'bert-base-uncased'  # Pre-trained BERT model name
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    # Tokenize the input text
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)

    # Convert the token IDs to PyTorch tensors
    input_ids = torch.tensor([token_ids])

    # Set the model to evaluation mode
    model.eval()

    # Vectorize the input text using BERT
    with torch.no_grad():
        encoded_layers, _ = model(input_ids)

    # Get the vector representation from the last BERT layer
    vectorized_text = encoded_layers[-1].squeeze(0)

    return vectorized_text


def bert_vector_cosine(text1, text2):
    vector1 = vectorize_with_bert(text1)
    vector2 = vectorize_with_bert(text2)

    # Calculate the cosine similarity
    similarity = cosine_similarity(vector1, vector2)

    return similarity[0][0], similarity[0][0], []
