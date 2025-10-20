#!/usr/bin/env python3

import argparse
import json
import math
import os
import pickle
import string
from collections import Counter
from nltk.stem import PorterStemmer

BM25_K1 = 1.5

BM25_B = 0.75

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
        self.doc_lengths = {}

    def __add_document(self, doc_id, text):
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        clean_text = text.lower().translate(translator)
        tokens = [stemmer.stem(t) for t in clean_text.split() if t and t not in self.stopwords]
        self.doc_lengths[doc_id] = len(tokens)
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
        with open('cache/doc_lengths.pkl', 'wb') as f:
            pickle.dump(self.doc_lengths, f)

    def load(self):
        if not (os.path.exists('cache/index.pkl') and os.path.exists('cache/docmap.pkl') and os.path.exists('cache/term_frequencies.pkl')):
            raise FileNotFoundError("Index files not found in cache directory")
        with open('cache/index.pkl', 'rb') as f:
            self.index = pickle.load(f)
        with open('cache/docmap.pkl', 'rb') as f:
            self.docmap = pickle.load(f)
        with open('cache/term_frequencies.pkl', 'rb') as f:
            self.term_frequencies = pickle.load(f)
        try:
            with open('cache/doc_lengths.pkl', 'rb') as f:
                self.doc_lengths = pickle.load(f)
        except FileNotFoundError:
            self.doc_lengths = {}

    def get_tf(self, doc_id, term):
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        clean_term = term.lower().translate(translator)
        tokens = [stemmer.stem(t) for t in clean_term.split() if t and t not in self.stopwords]
        if len(tokens) != 1:
            raise ValueError("Term must be a single token after processing")
        stemmed_term = tokens[0]
        return self.term_frequencies.get(doc_id, Counter()).get(stemmed_term, 0)

    def get_bm25_idf(self, term):
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        clean_term = term.lower().translate(translator)
        tokens = [stemmer.stem(t) for t in clean_term.split() if t and t not in self.stopwords]
        if len(tokens) != 1:
            raise ValueError("Term must be a single token after processing")
        stemmed_term = tokens[0]
        df = len(self.index.get(stemmed_term, set()))
        N = len(self.docmap)
        bm25_idf = math.log((N - df + 0.5) / (df + 0.5) + 1)
        return bm25_idf

    def __get_avg_doc_length(self) -> float:
        if not self.doc_lengths:
            return 0.0
        return sum(self.doc_lengths.values()) / len(self.doc_lengths)

    def get_bm25_tf(self, doc_id: int, term: str, k1: float = BM25_K1, b: float = BM25_B) -> float:
        tf = self.get_tf(doc_id, term)
        doc_len = self.doc_lengths.get(doc_id, 0)
        avg_dl = self.__get_avg_doc_length()
        if avg_dl == 0:
            return 0.0
        normalization = 1 - b + b * (doc_len / avg_dl)
        return (tf * (k1 + 1)) / (tf + k1 * normalization)

    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf

    def bm25_search(self, query, limit):
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        query_clean = query.lower().translate(translator)
        query_tokens = [stemmer.stem(t) for t in query_clean.split() if t and t not in self.stopwords]
        if not query_tokens:
            return []
        scores = {}
        for doc_id in self.docmap:
            total_score = 0.0
            for token in query_tokens:
                try:
                    score = self.bm25(doc_id, token)
                    total_score += score
                except ValueError:
                    pass
            scores[doc_id] = total_score
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [(doc_id, score) for doc_id, score in sorted_docs[:limit]]


def bm25_idf_command(term):
    index = InvertedIndex()
    index.load()
    return index.get_bm25_idf(term)


def bm25_tf_command(doc_id, term, k1=BM25_K1, b=BM25_B):
    index = InvertedIndex()
    index.load()
    return index.get_bm25_tf(doc_id, term, k1, b)


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
    bm25_idf_parser = subparsers.add_parser('bm25idf', help="Get BM25 IDF score for a given term")
    bm25_idf_parser.add_argument("term", type=str, help="Term to get BM25 IDF score for")

    bm25_tf_parser = subparsers.add_parser("bm25tf", help="Get BM25 TF score for a given document ID and term")
    bm25_tf_parser.add_argument("doc_id", type=int, help="Document ID")
    bm25_tf_parser.add_argument("term", type=str, help="Term to get BM25 TF score for")
    bm25_tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
    bm25_tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

    bm25search_parser = subparsers.add_parser("bm25search", help="Search movies using full BM25 scoring")
    bm25search_parser.add_argument("query", type=str, help="Search query")
    bm25search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return")

    args = parser.parse_args()

    if args.command == "build":
        with open('./data/movies.json', 'r') as f:
            movies = json.load(f)
        index = InvertedIndex()
        index.build(movies)
        index.save()
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
    elif args.command == "bm25idf":
        try:
            bm25idf = bm25_idf_command(args.term)
            print(f"BM25 IDF score of '{args.term}': {bm25idf:.2f}")
        except FileNotFoundError:
            print("Index not built. Please run 'build' command first.")
        except ValueError as e:
            print(str(e))
    elif args.command == "bm25tf":
        try:
            bm25tf = bm25_tf_command(args.doc_id, args.term, args.k1, args.b)
            print(f"BM25 TF score of '{args.term}' in document '{args.doc_id}': {bm25tf:.2f}")
        except FileNotFoundError:
            print("Index not built. Please run 'build' command first.")
        except ValueError as e:
            print(str(e))
    elif args.command == "bm25search":
        index = InvertedIndex()
        try:
            index.load()
        except FileNotFoundError:
            print("Index not built. Please run 'build' command first.")
            return
        results = index.bm25_search(args.query, args.limit)
        if results:
            for i, (doc_id, score) in enumerate(results, start=1):
                movie = index.docmap[doc_id]
                print(f"{i}. ({doc_id}) {movie['title']} - Score: {score:.2f}")
        else:
            print("No movies found.")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()