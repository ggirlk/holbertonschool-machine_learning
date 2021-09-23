#!/usr/bin/env python3
"""Display soonest upcoming launch"""


import requests


if __name__ == '__main__':
    req = requests.get('https://api.spacexdata.com/v4/launches/upcoming',
                       headers={'pagination': 'false'})
    json = req.json()
    time = 99999999999
    next = None
    for launch in json:
        thistime = int(launch['date_unix'])
        if thistime < time:
            time = thistime
            next = launch
    if next is not None:
        rocket = requests.get('https://api.spacexdata.com/v4/rockets/'
                              + next['rocket'])
        rocket = rocket.json()['name']
        launchpads = requests.get('https://api.spacexdata.com/v4/launchpads/'
                                  + next['launchpad'])
        launchpads = launchpads.json()
        locale = launchpads['locality']
        launchpads = launchpads['name']
        print('{} ({}) {} - {} ({})'.format(next['name'], next['date_local'],
                                            rocket, launchpads, locale))
