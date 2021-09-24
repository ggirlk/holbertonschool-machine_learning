#!/usr/bin/env python3
"""top students"""


def top_students(mongo_collection):
    """
    Function that returns all students sorted by average score
    @mongo_collection: will be the pymongo collection object
    """
    students = mongo_collection.find()
    best_students = []
    for student in students:
        topics = student["topics"]
        score = 0
        for topic in topics:
            score = score + topic["score"]
        avg = score / len(topics)
        student["averageScore"] = avg
        best_students.append(student)
    return sorted(best_students, key=lambda i: i["averageScore"], reverse=True)
