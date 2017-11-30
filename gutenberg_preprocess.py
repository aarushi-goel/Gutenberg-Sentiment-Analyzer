from gutenberg.acquire import load_etext
from gutenberg.cleanup import strip_headers
from gutenberg._domain_model.exceptions import UnknownDownloadUriException
from gutenberg.query import get_etexts
from gutenberg.query import get_metadata
from Book import Book
import pickle


def trial():
    text = strip_headers(load_etext(2701)).strip()
    print(text)  # prints 'MOBY DICK; OR THE WHALE\n\nBy Herman Melville ...'
    print(get_metadata('title', 2701))  # prints frozenset([u'Moby Dick; Or, The Whale'])
    print(get_metadata('author', 2701)) # prints frozenset([u'Melville, Hermann'])

    print(get_etexts('title', 'Moby Dick; Or, The Whale'))  # prints frozenset([2701, ...])
    print(get_etexts('author', 'Melville, Herman'))        # prints frozenset([2701, ...])


def init_books(author_file, json_file):
    """initialize book list with texts and save it to disk"""
    with open(author_file) as f:
        authors = list(f)

    authors = [i.strip() for i in authors]

    books = []
    for author in authors:
        s = get_etexts('author', author)
        for i in s:
            try:
                if list(get_metadata('language', i))[0] == 'en':
                    title, etext = list(get_metadata('title', i))[0], strip_headers(load_etext(i)).strip()
                    b = Book(i, title, etext)
                    books.append(b)
            except UnknownDownloadUriException:
                # this book does not have a load_etext corresponding to it.
                pass

    with open(json_file, 'wb') as f:
        pickle.dump(books, f)

    print (len(books))


def main():
    init_books(author_file='author.txt', json_file='data')


if __name__ == '__main__':
    main()
