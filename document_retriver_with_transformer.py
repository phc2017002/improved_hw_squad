import json
import numpy as np
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from unidecode import unidecode


#nltk.download('stopwords')
#nltk.download('wordnet')

def preprocess_text(text):
    # Lowercase the text
    text = text.lower()
    # Define the punctuation to keep (hyphen is included)
    keep_punctuation = ['-']
    # Define the punctuation to remove (excluding hyphen)
    remove_punctuation = string.punctuation.translate(str.maketrans('', '', ''.join(keep_punctuation)))
    # Remove punctuation
    translator = str.maketrans('', '', remove_punctuation)
    text = text.translate(translator)
    return text

# Load the JSON data
with open('/ssd_scratch/cvit/ani31101993/BERT_baseline/BenthamQA_Squad_like_200.json', 'r') as json_file:
    data = json.load(json_file)


print(len(data))
# Extract all contexts from the data
all_contexts = [preprocess_text(unidecode(entry['contexts'])) for entry in data]


contexts = []
queries_and_contexts = {}

for entry in data:
    context = preprocess_text(unidecode(entry['contexts']))
    #context = preprocess_text(context)
    contexts.append(context)
    for qa in entry['qas']:
        query = preprocess_text(unidecode(qa['question']))
        #query = preprocess_text(query)
        queries_and_contexts[query] = context

vectorizers = [
    TfidfVectorizer(sublinear_tf=True, smooth_idf=True),
    CountVectorizer()
]
model = SentenceTransformer('all-MiniLM-L6-v2')

# Fit and transform the contexts using each vectorizer
matrices = [vectorizer.fit_transform(contexts) for vectorizer in vectorizers]

# Encode all contexts once using the Sentence Transformer
context_embeddings = model.encode(contexts)

# Function to retrieve top 5 documents
def retrieve_top_documents_ensemble(query, matrices, vectorizers, context_embeddings, n=5):
    similarity_scores = []
    for i, vectorizer in enumerate(vectorizers):
        query_vector = vectorizer.transform([query])
        cosine_similarities = cosine_similarity(query_vector, matrices[i]).flatten()
        similarity_scores.append(cosine_similarities)
    
    query_embedding = model.encode([query])
    transformer_similarities = cosine_similarity(query_embedding.reshape(1, -1), context_embeddings).flatten()
    similarity_scores.append(transformer_similarities)
    
    weights = [0.6, 0.0, 0.3]
    weighted_scores = np.average(similarity_scores, axis=0, weights=weights)
    
    top_n_indices = weighted_scores.argsort()[-n:][::-1]
    return [contexts[i] for i in top_n_indices]

new_data = []



# Process the data and update the JSON structure
for entry in data:
    # Extract the original context and questions
    original_context = preprocess_text(unidecode(entry['contexts']))
    questions = entry['qas']
    
    # Replace the original context with the top 5 contexts for each question
    all_information = {}
    for qa in questions:
        #print('qa--------',qa)
        query_id = qa['id']
        query = preprocess_text(unidecode(qa['question']))
        #print('answers--------', qa['answers'])
        #for answer in qa['answers']:
        #    answer['text'] = preprocess_text(answer['text'])

        #answers = [{'text': preprocess_text(unidecode(answer['text']))} for answer in qa['answers']]

        #answers = [{'text': preprocess_text(unidecode(answer['text']))} for answer in qa['answers']]

        #print('answers after preprocess--------', qa['answers'])
        answer = preprocess_text(unidecode(qa['answers'][0]['text']))
        #answer_start = qa['answers'][0]['answer_start']
        # Retrieve the top 5 contexts for the current query
        #all_contexts = unidecode(all_contexts)
        #all_contexts = preprocess_text(all_contexts)
        top_5_contexts = retrieve_top_documents_ensemble(query, matrices, vectorizers, context_embeddings)
        # Update the JSON structure with the top 5 contexts
        #qa['top_5_contexts'] = top_5_contexts


        #print(top_5_contexts)
        combined_context = ' '.join(top_5_contexts)
        query = preprocess_text(unidecode(query))
        #answer = unidecode(answer)
        #top_5_contexts = preprocess_text(top_5_contexts)
        #query = preprocess_text(query)
        #answer = preprocess_text(answer)
        #for answer in answers:
        #    answer['answer_start'] = -1


        answer_start = combined_context.find(answer)

        if answer_start == -1:
            continue
        
        all_information = {
                  'contexts' : combined_context,
                  'qas': [{'id': query_id, 'question': query, 'answers': [{'text': answer, 'answer_start': answer_start}]}]   
              }
        
        new_data.append(all_information)

# [{'text': answer, 'answer_start': answer_start}]
def compute_top_5_accuracy(data):
    total_questions = 0
    correct_top_5 = 0

    for entry in data:
        for qa in entry['qas']:
            total_questions += 1
            query = preprocess_text(unidecode(qa['question']))
            answer = preprocess_text(unidecode(qa['answers'][0]['text']))

            top_5_contexts = retrieve_top_documents_ensemble(query, matrices, vectorizers, context_embeddings, n=5)
            
            if any(answer in context for context in top_5_contexts):
                correct_top_5 += 1

    top_5_accuracy = correct_top_5 / total_questions
    return top_5_accuracy

# Compute top-5 accuracy
top_5_accuracy = compute_top_5_accuracy(new_data)
print(f"Top-5 Accuracy: {top_5_accuracy:.4f}")


print(len(new_data))
print(new_data[0:2])    


# Save the updated JSON data back to the file
with open('/ssd_scratch/cvit/ani31101993/BERT_baseline/BenthamQA_Squad_like_tf_idf_with_transformer_200_qa.json', 'w') as json_file:
    json.dump(new_data, json_file, indent=4)
