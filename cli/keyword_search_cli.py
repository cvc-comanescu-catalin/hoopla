#!/usr/bin/env python3

import argparse
import json
import math
import os
import pickle
import string
from collections import Counter
from nltk.stem import PorterStemmer


class InvertedIndex:
    def __init__(self, stopwords=None):
        if stopwords is None:
            with open('./data/stopwords.txt', 'r') as f:
                self.stopwords = set(line.strip().lower() for line in f if line.strip())
        else:
            self.stopwords = stopwords
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}

    def __add_document(self, doc_id, text):
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        clean_text = text.lower().translate(translator)
        tokens = [stemmer.stem(t) for t in clean_text.split() if t and t not in self.stopwords]
        if doc_id not in self.term_frequencies:
            self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)
            self.term_frequencies[doc_id][token] += 1

    def get_documents(self, term):
        term = term.lower()
        return sorted(list(self.index.get(term, set())))

    def build(self, movies):
        for movie in movies["movies"]:
            doc_id = movie["id"]
            text = f"{movie['title']} {movie['description']}"
            self.__add_document(doc_id, text)
            self.docmap[doc_id] = movie

    def save(self):
        os.makedirs('cache', exist_ok=True)
        with open('cache/index.pkl', 'wb') as f:
            pickle.dump(self.index, f)
        with open('cache/docmap.pkl', 'wb') as f:
            pickle.dump(self.docmap, f)
        with open('cache/term_frequencies.pkl', 'wb') as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        if not (os.path.exists('cache/index.pkl') and os.path.exists('cache/docmap.pkl') and os.path.exists('cache/term_frequencies.pkl')):
            raise FileNotFoundError("Index files not found in cache directory")
        with open('cache/index.pkl', 'rb') as f:
            self.index = pickle.load(f)
        with open('cache/docmap.pkl', 'rb') as f:
            self.docmap = pickle.load(f)
        with open('cache/term_frequencies.pkl', 'rb') as f:
            self.term_frequencies = pickle.load(f)

    def get_tf(self, doc_id, term):
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        clean_term = term.lower().translate(translator)
        tokens = [stemmer.stem(t) for t in clean_term.split() if t and t not in self.stopwords]
        if len(tokens) != 1:
            raise ValueError("Term must be a single token after processing")
        stemmed_term = tokens[0]
        return self.term_frequencies.get(doc_id, Counter()).get(stemmed_term, 0)


def main() -> None:
    with open('./data/stopwords.txt', 'r') as f:
        stopwords = set(line.strip().lower() for line in f if line.strip())
    
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    search_parser = subparsers.add_parser("search", help="Search movies using inverted index")
    search_parser.add_argument("query", type=str, help="Search query")
    tf_parser = subparsers.add_parser("tf", help="Get term frequency for a document")
    tf_parser.add_argument("doc_id", type=int, help="Document ID")
    tf_parser.add_argument("term", type=str, help="Term")
    idf_parser = subparsers.add_parser("idf", help="Calculate IDF for a term")
    idf_parser.add_argument("term", type=str, help="Term")
    tfidf_parser = subparsers.add_parser("tfidf", help="Calculate TF-IDF for a document and term")
    tfidf_parser.add_argument("doc_id", type=int, help="Document ID")
    tfidf_parser.add_argument("term", type=str, help="Term")

    args = parser.parse_args()

    if args.command == "build":
        with open('./data/movies.json', 'r') as f:
            movies = json.load(f)
        index = InvertedIndex()
        index.build(movies)
        index.save()
        merida_docs = index.get_documents('merida')
        if merida_docs:
            print(f"First document for token 'merida' = {merida_docs[0]}")
        else:
            print("No documents for 'merida'")
    elif args.command == "search":
        print(f"Searching for: {args.query}")
        index = InvertedIndex()
        try:
            index.load()
        except FileNotFoundError:
            print("Index not built. Please run 'build' command first.")
            return
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        query_clean = args.query.lower().translate(translator)
        query_tokens = [stemmer.stem(t) for t in query_clean.split() if t and t not in index.stopwords]
        if not query_tokens:
            print("No valid query tokens after processing.")
            return
        # Get sets of doc_ids for each token
        doc_sets = [set(index.get_documents(token)) for token in query_tokens]
        if doc_sets:
            # Union all sets
            result_ids = set.union(*doc_sets) if doc_sets else set()
        else:
            result_ids = set()
        results = [index.docmap[doc_id] for doc_id in sorted(result_ids)][:5]
        if results:
            print("Found movies:")
            for i, movie in enumerate(results, start=1):
                print(f"{i}. {movie['title']}")
        else:
            print("No movies found.")
    elif args.command == "tf":
        index = InvertedIndex()
        try:
            index.load()
        except FileNotFoundError:
            print("Index not built. Please run 'build' command first.")
            return
        try:
            freq = index.get_tf(args.doc_id, args.term)
            print(freq)
        except ValueError as e:
            print(str(e))
    elif args.command == "idf":
        index = InvertedIndex()
        try:
            index.load()
        except FileNotFoundError:
            print("Index not built. Please run 'build' command first.")
            return
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        clean_term = args.term.lower().translate(translator)
        tokens = [stemmer.stem(t) for t in clean_term.split() if t and t not in index.stopwords]
        if len(tokens) != 1:
            print("Term must be a single token after processing")
            return
        stemmed_term = tokens[0]
        term_doc_count = len(index.index.get(stemmed_term, set()))
        doc_count = len(index.docmap)
        idf = math.log((doc_count + 1) / (term_doc_count + 1))
        print(f"Inverse document frequency of '{args.term}': {idf:.2f}")
    elif args.command == "tfidf":
        index = InvertedIndex()
        try:
            index.load()
        except FileNotFoundError:
            print("Index not built. Please run 'build' command first.")
            return
        try:
            tf = index.get_tf(args.doc_id, args.term)
        except ValueError as e:
            print(str(e))
            return
        # Now compute IDF
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        clean_term = args.term.lower().translate(translator)
        tokens = [stemmer.stem(t) for t in clean_term.split() if t and t not in index.stopwords]
        if len(tokens) != 1:
            print("Term must be a single token after processing")
            return
        stemmed_term = tokens[0]
        term_doc_count = len(index.index.get(stemmed_term, set()))
        doc_count = len(index.docmap)
        idf = math.log((doc_count + 1) / (term_doc_count + 1))
        tf_idf = tf * idf
        print(f"TF-IDF score of '{args.term}' in document '{args.doc_id}': {tf_idf:.2f}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()