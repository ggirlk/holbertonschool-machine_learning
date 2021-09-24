#!/usr/bin/env python3
"""Method List all"""


def list_all(mongo_collection):
    """
    Method that lists all documents in a collection:
    @mongo_collection: pymongo collection object
    Return:
        empty list if no document in the collection
    """
    documents = []
    collection = mongo_collection.find()
    for doc in collection:
        documents.append(doc)
    return documents
