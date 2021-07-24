#!/usr/bin/env python3
""" Swapi API """
import requests


def availableShips(passengerCount):
    """
    *********************************************
    **********returns the list of ships**********
    *********************************************
    @passengerCount: a given number of passengers
    Return:
           List of ships or empty list 
    """
    url = "https://swapi-api.hbtn.io/api/starships"
    r = requests.get(url)
    json = r.json()
    ships = []
    while r.status_code == 200:
        res = json['results']
        for ship in res:
            if (ship['passengers'] >= str(passengerCount)):
                ships.append(ship['name'])

        url = json["next"]
        if (url is not None):
            r = requests.get(url)
            json = r.json()
        else:
            break

    return ships
