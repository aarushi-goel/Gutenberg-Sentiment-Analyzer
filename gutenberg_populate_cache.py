def populate_cache():
    """ Populates metadata cache to work with gutenberg api"""
    from gutenberg.acquire import get_metadata_cache
    cache = get_metadata_cache()
    cache.populate()


if __name__ == '__main__':
    populate_cache()