#!/usr/bin/env python3

import argparse
import json
import os
import pickle
import string
from nltk.stem import PorterStemmer


class InvertedIndex:
    def __init__(self, stopwords):
        self.index = {}
        self.docmap = {}
        self.stopwords = stopwords

    def __add_document(self, doc_id, text):
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        clean_text = text.lower().translate(translator)
        tokens = [stemmer.stem(t) for t in clean_text.split() if t and t not in self.stopwords]
        for token in tokens:
            if token not in self.index:
                self.index[token] = set()
            self.index[token].add(doc_id)

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

    def load(self):
        if not os.path.exists('cache/index.pkl') or not os.path.exists('cache/docmap.pkl'):
            raise FileNotFoundError("Index files not found in cache directory")
        with open('cache/index.pkl', 'rb') as f:
            self.index = pickle.load(f)
        with open('cache/docmap.pkl', 'rb') as f:
            self.docmap = pickle.load(f)


def main() -> None:
    with open('./data/stopwords.txt', 'r') as f:
        stopwords = set(line.strip().lower() for line in f if line.strip())
    
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    build_parser = subparsers.add_parser("build", help="Build the inverted index")
    search_parser = subparsers.add_parser("search", help="Search movies using inverted index")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    if args.command == "build":
        with open('./data/movies.json', 'r') as f:
            movies = json.load(f)
        index = InvertedIndex(stopwords)
        index.build(movies)
        index.save()
        merida_docs = index.get_documents('merida')
        if merida_docs:
            print(f"First document for token 'merida' = {merida_docs[0]}")
        else:
            print("No documents for 'merida'")
    elif args.command == "search":
        print(f"Searching for: {args.query}")
        index = InvertedIndex(None)
        try:
            index.load()
        except FileNotFoundError:
            print("Index not built. Please run 'build' command first.")
            return
        translator = str.maketrans('', '', string.punctuation)
        stemmer = PorterStemmer()
        query_clean = args.query.lower().translate(translator)
        query_tokens = [stemmer.stem(t) for t in query_clean.split() if t and t not in stopwords]
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
    else:
        parser.print_help()


if __name__ == "__main__":
    main()