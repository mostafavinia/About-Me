import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

def calculate_tf_idf(corpus):
    vectorizer = TfidfVectorizer()
    tf_idf_matrix = vectorizer.fit_transform(corpus)
    return tf_idf_matrix

def calculate_tf(corpus):
    vectorizer = TfidfVectorizer(use_idf=False)
    tf_matrix = vectorizer.fit_transform(corpus)
    return tf_matrix

def process_folder(folder_path):
    categories = []
    corpus = []
    file_paths = []
    
    for root, dirs, files in os.walk(folder_path):
        category = os.path.basename(root)
        categories.extend([category] * len(files))
        
        for file in files:
            file_path = os.path.join(root, file)
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
                corpus.append(text)
                file_paths.append(file_path)
    
    return categories, corpus, file_paths

def categorize_query(query, categories, corpus, file_paths):
    query_corpus = [query] * len(categories)
    
    tf_idf_matrix = calculate_tf_idf(corpus + query_corpus)
    tf_matrix = calculate_tf(corpus + query_corpus)
    
    similarity_scores = cosine_similarity(tf_idf_matrix[-1], tf_idf_matrix[:-1])
    tf_scores = cosine_similarity(tf_matrix[-1], tf_matrix[:-1])
    
    results = []
    for category, score, tf_score, file_path in zip(categories, similarity_scores[0], tf_scores[0], file_paths):
        results.append((category, score, tf_score, file_path))
    
    results = sorted(results, key=lambda x: x[1], reverse=True)
    
    return results

# Example usage
folder_path = './Data'

categories, corpus, file_paths = process_folder(folder_path)

query = "دانشمند مطرح در زمینه سلولهای بنیادی"

results = categorize_query(query, categories, corpus, file_paths)

for category, score, tf_score, file_path in results:
    print(f"دسته بندی = {category}")
    print(f"TF-IDF Score= {score}")
    print(f"TF Score= {tf_score}")
    print(f"File Path= {file_path}")
    print()
