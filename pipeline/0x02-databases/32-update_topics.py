#!/usr/bin/env python3
"""Method update topics"""


def update_topics(mongo_collection, name, topics):
    """
    function that changes all topics of a school document based
        on the name
    @mongo_collection: pymongo collection object
    @name: (string) school name to update
    @topics: (list of strings) list of topics approached
            in the school
    """
    vals = {"$set": {"topics": topics}}
    mongo_collection.update_many({"name": name}, vals)
