#!/usr/bin/env python3

import argparse
import json


def main() -> None:
    with open('./data/movies.json', 'r') as f:
        movies = json.load(f)
    
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()

    match args.command:
        case "search":
            # print the search query here
            print(f"Searching for: {args.query}")
            results = []
            for movie in movies["movies"]:
                if args.query.lower() in movie["title"].lower():
                    results.append(movie)
            results = sorted(results, key=lambda m: m['id'])[:5]
            if results:
                print("Found movies:")
                for i, movie in enumerate(results, start=1):
                    print(f"{i}. {movie['title']}")
            else:
                print("No movies found.")
            pass
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()