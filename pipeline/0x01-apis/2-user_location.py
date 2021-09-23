#!/usr/bin/env python3
"""
prints the location of a specific user
"""
import requests
import sys
import time


if __name__ == '__main__':

    if len(sys.argv) == 2:
        url = sys.argv[1]
        r = requests.get(url)

        if r.status_code == 404:
            print("Not found")
        if r.status_code == 403:
            reset = r.headers['X-Ratelimit-Reset']
            reset = int(reset) - int(time.time())
            print("Reset in {} min".format(int(reset / 60)))
        else:
            print("Not found")
        if r.status_code == 200:
            json = r.json()
            print(json["location"])
        
