#!/usr/bin/env python3
"""Display number of launches per rocket"""


import requests


if __name__ == '__main__':
    req = requests.get('https://api.spacexdata.com/v4/launches',
                        headers={'pagination': 'false'})
    rockets = {}
    for launch in req.json():
        rocket = launch['rocket']
        rockets[rocket] = rockets.get(rocket, 0) + 1
    rocket_sort = []
    launches = []
    for rocket in rockets:
        name = requests.get('https://api.spacexdata.com/v4/rockets/'
                            + rocket).json()['name']
        launches.append(rockets[rocket])
        rocket_sort.append(name)
    rsl = zip(rocket_sort, launches)
    rsl = list(rsl)
    rsl.sort(key=lambda x: x[0])
    rsl.sort(key=lambda x: x[1], reverse=True)
    for pair in rsl:
        print("{}: {}".format(pair[0], pair[1]))
