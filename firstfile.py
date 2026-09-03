import glob
import math
import os
import xml.etree.ElementTree as ET
from collections import defaultdict, Counter
import pandas as pd
import re
from parsivar import Normalizer, Tokenizer, FindStems
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import openpyxl
import tkinter as tk
from tkinter import ttk
from pathlib import Path

base_dir = Path(__file__).resolve().parent

folder_path = base_dir / "Dataset"
stop_words_file = base_dir / "stopwords.xml"
category_mapping_file = base_dir / "category_mapping.txt"

def load_category_mapping(file_path):
    mapping = {}
    with open(file_path, 'r', encoding='utf-16') as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split('\t')
                if len(parts) >= 3:
                    code = parts[0].strip()
                    eng = parts[1].strip()
                    farsi = parts[2].strip()
                    mapping[code] = {'english': eng, 'farsi': farsi}
                elif len(parts) >= 2:
                    code = parts[0].strip()
                    eng = parts[1].strip()
                    mapping[code] = {'english': eng, 'farsi': None}
    return mapping

def parse_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    data = []
    for doc in root.findall('DOC'):
        doc_id = doc.find('DOCID').text if doc.find('DOCID') is not None else ""
        doc_no = doc.find('DOCNO').text if doc.find('DOCNO') is not None else ""
        title = doc.find('TITLE').text if doc.find('TITLE') is not None else ""
        text = doc.find('TEXT').text if doc.find('TEXT') is not None else ""
        category = ""
        cat_tags = doc.findall('CAT')
        if cat_tags:
            category = cat_tags[0].text.strip() if cat_tags[0].text else ""
        data.append({
            'DOCID': doc_id.strip(),
            'DOCNO': doc_no.strip(),
            'Title': title.strip(),
            'Text': text.strip(),
            'Category': category
        })
    return data

def load_stop_words_from_xml(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()
    stop_words = [word.text.strip() for word in root.findall('Word')]
    return set(stop_words)

def save_to_excel(data, file_path, sheet_name, columns):
    df = pd.DataFrame(data, columns=columns)
    with pd.ExcelWriter(file_path, engine='openpyxl', mode='w') as writer:
        df.to_excel(writer, index=False, sheet_name=sheet_name)
    print(f'Data saved to {file_path} ({sheet_name})')

def build_inverted_index(raw_documents):
    inverted_index = defaultdict(list)
    for doc_id, tokens in raw_documents:
        for token in tokens:
            if doc_id not in inverted_index[token]:
                inverted_index[token].append(doc_id)
    return inverted_index

def clean_data_stemmed(documents, stop_words):
    stemmer = FindStems()
    cleaned_documents = []
    for doc_id, tokens in documents:
        new_tokens = []
        for token in tokens:
            token = re.sub(r'[^آ-یa-zA-Z]+', '', token)
            token = token.strip()
            if token and (token not in stop_words):
                new_tokens.append(token)
        stemmed_tokens = [stemmer.convert_to_stem(tk) for tk in new_tokens if tk.strip()]
        cleaned_documents.append((doc_id, stemmed_tokens))
    return cleaned_documents

def compute_tf(tokens):
    tf_counter = Counter(tokens)
    return dict(tf_counter)

def compute_idf(documents):
    N = len(documents)
    idf = {}
    all_tokens_set = set(token for doc_id, tokens in documents for token in tokens)
    for token in all_tokens_set:
        df = sum(1 for _, tokens in documents if token in tokens)
        idf[token] = math.log((N + 1) / (1 + df)) + 1
    return idf

def compute_tf_idf(documents):
    idf_values = compute_idf(documents)
    tf_idf_docs = []
    for doc_id, tokens in documents:
        tf_values = compute_tf(tokens)
        tf_idf = {word: tf_values[word] * idf_values[word] for word in tf_values}
        tf_idf_docs.append((doc_id, tf_idf))
    return tf_idf_docs

def compute_document_similarity(tf_idf_documents):
    doc_ids = [doc_id for doc_id, _ in tf_idf_documents]
    vocabulary = sorted(set(token for _, tf_dict in tf_idf_documents for token in tf_dict.keys()))
    tf_idf_matrix = []
    for _, tf_dict in tf_idf_documents:
        row_vector = [tf_dict.get(token, 0) for token in vocabulary]
        tf_idf_matrix.append(row_vector)
    tf_idf_matrix = np.array(tf_idf_matrix)
    similarity_matrix = cosine_similarity(tf_idf_matrix)
    similarity_df = pd.DataFrame(similarity_matrix, index=doc_ids, columns=doc_ids)
    return similarity_df, vocabulary, tf_idf_matrix, doc_ids

def compute_bm25_idf(documents):
    N = len(documents)
    df = {}
    for doc_id, tokens in documents:
        for token in set(tokens):
            df[token] = df.get(token, 0) + 1
    bm25_idf = {}
    for token, d in df.items():
        bm25_idf[token] = math.log((N - d + 0.5) / (d + 0.5) + 1)
    return bm25_idf

def retrieve_documents_bm25(query_tokens, docs, bm25_idf, k1=1.2, b=0.75):
    scores = {}
    doc_lengths = {doc_id: len(tokens) for doc_id, tokens in docs}
    avgdl = sum(doc_lengths.values()) / len(doc_lengths)
    query_tf = compute_tf(query_tokens)
    for doc_id, tokens in docs:
        score = 0
        doc_tf = compute_tf(tokens)
        for term in query_tf:
            if term in doc_tf:
                f = doc_tf[term]
                idf_val = bm25_idf.get(term, 0)
                score += idf_val * (f * (k1 + 1)) / (f + k1 * (1 - b + b * (doc_lengths[doc_id] / avgdl)))
        scores[doc_id] = score
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return ranked

def is_category_match(mapped_category, expected_category):
    return (expected_category.lower() in mapped_category.lower()) or (
            mapped_category.lower() in expected_category.lower())

def evaluate_query_by_category_bm25(query_text, expected_category, stop_words, bm25_idf, docs, doc_categories,
                                      category_mapping, k1=1.2, b=0.75):
    query_tokens = process_query(query_text, stop_words)
    ranked = retrieve_documents_bm25(query_tokens, docs, bm25_idf, k1, b)
    ranked_doc_ids = [doc_id for doc_id, score in ranked]
    total_retrieved = len(ranked_doc_ids)

    correct = 0
    sum_prec = 0.0
    rel_count = 0
    for i, doc in enumerate(ranked_doc_ids, start=1):
        doc_code = doc_categories.get(doc, "")
        mapped_category = category_mapping.get(doc_code, {}).get('farsi', doc_code)
        if is_category_match(mapped_category, expected_category):
            correct += 1
            rel_count += 1
            sum_prec += rel_count / i
    precision = correct / total_retrieved if total_retrieved > 0 else 0

    total_relevant = 0
    for doc, code in doc_categories.items():
        mapped_category = category_mapping.get(code, {}).get('farsi', code)
        if is_category_match(mapped_category, expected_category):
            total_relevant += 1
    recall = correct / total_relevant if total_relevant > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    avg_precision = sum_prec / total_relevant if total_relevant > 0 else 0

    return precision, recall, f1, avg_precision, ranked_doc_ids, correct, total_retrieved

def process_query(query_text, stop_words):
    normalizer = Normalizer()
    tokenizer = Tokenizer()
    stemmer = FindStems()
    normalized_q = normalizer.normalize(query_text)
    tokens = tokenizer.tokenize_words(normalized_q)
    cleaned_tokens = []
    for token in tokens:
        token = re.sub(r'[^آ-یa-zA-Z]+', '', token)
        token = token.strip()
        if token and (token not in stop_words):
            cleaned_tokens.append(token)
    stemmed = [stemmer.convert_to_stem(tk) for tk in cleaned_tokens if tk.strip()]
    return stemmed

def compute_query_vector(query_tokens, idf_dict):
    tf_q = compute_tf(query_tokens)
    query_tf_idf = {}
    for token, freq in tf_q.items():
        idf_val = idf_dict.get(token, 0.0)
        query_tf_idf[token] = freq * idf_val
    return query_tf_idf

def retrieve_documents_for_query(query_tf_idf, vocabulary, doc_tf_idf_matrix, doc_ids):
    query_vector = []
    for token in vocabulary:
        query_vector.append(query_tf_idf.get(token, 0.0))
    query_vector = np.array(query_vector).reshape(1, -1)
    similarities = cosine_similarity(query_vector, doc_tf_idf_matrix)
    sim_list = similarities[0]
    scored_docs = list(zip(doc_ids, sim_list))
    scored_docs.sort(key=lambda x: x[1], reverse=True)
    ranked_docs = [doc_id for doc_id, score in scored_docs]
    return ranked_docs, scored_docs

def launch_gui(categories, stop_words, bm25_idf, docs, doc_ids, doc_categories, category_mapping):
    root = tk.Tk()
    root.title("Category-Based Evaluation (BM25)")

    label = tk.Label(root, text="Select Category:")
    label.pack(pady=5)

    category_var = tk.StringVar()
    combobox = ttk.Combobox(root, textvariable=category_var, values=categories, state="readonly", width=40)
    combobox.pack(pady=5)
    combobox.current(0)

    text_widget = tk.Text(root, height=20, width=80)
    text_widget.pack(pady=5)

    def evaluate_selected():
        selected_category = category_var.get()
        precision, recall, f1, avg_precision, ranked_docs, correct, total_docs = evaluate_query_by_category_bm25(
            selected_category, selected_category, stop_words,
            bm25_idf, docs, doc_categories, category_mapping)
        result_text = f"Evaluation for query '{selected_category}' (expected category '{selected_category}'):\n"
        result_text += f"Total retrieved docs: {total_docs}\n"
        result_text += f"Retrieved docs: {ranked_docs}\n"
        result_text += f"Number of docs in expected category: {correct}\n"
        result_text += f"Precision: {precision:.4f}\n"
        result_text += f"Recall: {recall:.4f}\n"
        result_text += f"F1 Score: {f1:.4f}\n"
        result_text += f"MAP (Average Precision): {avg_precision:.4f}\n"
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, result_text)

    button1 = tk.Button(root, text="Evaluate Selected Category", command=evaluate_selected)
    button1.pack(pady=5)

    def evaluate_all():
        all_results = []
        for cat in categories:
            precision, recall, f1, avg_precision, ranked_docs, correct, total_docs = evaluate_query_by_category_bm25(
                cat, cat, stop_words,
                bm25_idf, docs, doc_categories, category_mapping)
            all_results.append((cat, total_docs, correct, precision, recall, f1, avg_precision))
        eval_columns = ["Category", "Total Retrieved", "Correct", "Precision", "Recall", "F1 Score", "MAP"]
        # ذخیره در پوشه اکسل (استفاده از excel_folder)
        eval_file = os.path.join(str(folder_path / "ExcelFiles"), "EvaluationResults_All.xlsx")
        save_to_excel(all_results, eval_file, "Evaluation", eval_columns)
        result_text = "Evaluation Results for All Categories:\n"
        for res in all_results:
            result_text += (f"Category: {res[0]}, Total: {res[1]}, Correct: {res[2]}, "
                            f"Precision: {res[3]:.4f}, Recall: {res[4]:.4f}, F1: {res[5]:.4f}, MAP: {res[6]:.4f}\n")
        text_widget.delete(1.0, tk.END)
        text_widget.insert(tk.END, result_text)

    button2 = tk.Button(root, text="Evaluate All Categories", command=evaluate_all)
    button2.pack(pady=5)

    root.mainloop()

def main():
    excel_folder = folder_path / "ExcelFiles"
    os.makedirs(excel_folder, exist_ok=True)

    stop_words = load_stop_words_from_xml(str(stop_words_file))

    category_mapping = load_category_mapping(str(category_mapping_file))

    farsi_categories = set()
    for mapping in category_mapping.values():
        if mapping.get('farsi'):
            farsi_categories.add(mapping['farsi'])
    farsi_categories = sorted(list(farsi_categories))

    files = glob.glob(str(folder_path / "*.xml"), recursive=True)
    documents_for_inverted = []
    doc_categories = {}
    normalizer = Normalizer()
    tokenizer = Tokenizer()

    for file_path_xml in files:
        file_data = parse_xml(file_path_xml)
        for item in file_data:
            doc_no = item['DOCNO']
            text = item['Text']
            cat = item['Category']
            doc_categories[doc_no] = cat
            raw_normalized = normalizer.normalize(text)
            raw_tokens = tokenizer.tokenize_words(raw_normalized)
            documents_for_inverted.append((doc_no, raw_tokens))
    print(f"Total Documents: {len(documents_for_inverted)}")

    inverted_index = build_inverted_index(documents_for_inverted)
    inverted_index_file = os.path.join(str(excel_folder), "InvertedIndex.xlsx")
    save_to_excel([(token, str(doc_ids)) for token, doc_ids in inverted_index.items()],
                  inverted_index_file, "Inverted Index", ["Token", "Documents"])

    docs_stemmed = clean_data_stemmed(documents_for_inverted, stop_words)
    cleaned_docs_file = os.path.join(str(excel_folder), "CleanedDocs.xlsx")
    save_to_excel([(doc_id, str(tokens)) for doc_id, tokens in docs_stemmed],
                  cleaned_docs_file, "Cleaned Documents", ["DOCNO", "Cleaned Tokens"])

    tf_data = []
    idf_dict_temp = compute_idf(docs_stemmed)
    for doc_id, tokens in docs_stemmed:
        freq_dict = compute_tf(tokens)
        for token, count in freq_dict.items():
            tf_data.append((doc_id, token, count, idf_dict_temp.get(token, 0.0)))
    tf_file = os.path.join(str(excel_folder), "TF.xlsx")
    save_to_excel(tf_data, tf_file, "TF", ["DOCNO", "Token", "TF", "IDF"])

    tf_idf_docs = compute_tf_idf(docs_stemmed)
    tf_idf_file = os.path.join(str(excel_folder), "TF_IDF.xlsx")
    save_to_excel([(doc_id, str(tf_idf)) for doc_id, tf_idf in tf_idf_docs],
                  tf_idf_file, "TF-IDF", ["DOCNO", "TF-IDF"])

    similarity_df, vocabulary, doc_tf_idf_matrix, doc_ids_tf = compute_document_similarity(tf_idf_docs)
    sim_cols = similarity_df.reset_index().columns.tolist()
    sim_file = os.path.join(str(excel_folder), "SimilarityMatrix.xlsx")
    save_to_excel(similarity_df.reset_index(), sim_file, "Similarity Matrix", sim_cols)
    print("Similarity matrix saved to", sim_file)

    bm25_idf = compute_bm25_idf(docs_stemmed)
    bm25_doc_ids = [doc_id for doc_id, tokens in docs_stemmed]

    query_text = "ورزشی"
    expected_category = "ورزشی"
    precision, recall, f1, avg_precision, ranked_docs, correct, total_docs = evaluate_query_by_category_bm25(
        query_text, expected_category, stop_words,
        bm25_idf, docs_stemmed, doc_categories, category_mapping)
    print("=== Console Evaluation ===")
    print(f"Evaluation for query '{query_text}' with expected category '{expected_category}':")
    print(f"Total retrieved docs: {total_docs}")
    print(f"Retrieved docs: {ranked_docs}")
    print(f"Number of docs in expected category: {correct}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print(f"MAP (Average Precision): {avg_precision:.4f}")

    evaluation_results = [
        (query_text, total_docs, correct, precision, recall, f1, avg_precision)
    ]
    eval_columns = ["Query", "Total Retrieved", "Correct", "Precision", "Recall", "F1 Score", "MAP"]
    eval_file = os.path.join(str(excel_folder), "EvaluationResults.xlsx")
    save_to_excel(evaluation_results, eval_file, "Evaluation", eval_columns)

    launch_gui(farsi_categories, stop_words, bm25_idf, docs_stemmed, bm25_doc_ids, doc_categories, category_mapping)

if __name__ == '__main__':
    main()
