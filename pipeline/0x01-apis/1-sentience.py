#!/usr/bin/env python3
""" Swapi API """
import requests


def sentientPlanets():
    """
     Return:
             the list of names of the home
             planets of all sentient species
    """

    url = "https://swapi-api.hbtn.io/api/species"
    r = requests.get(url)
    json = r.json()
    planets = []
    while r.status_code == 200:
        res = json['results']
        for spec in res:
            if spec["designation"] == "sentient":

                urlh = spec["homeworld"]

                if (urlh is not None):
                    rh = requests.get(urlh)
                    jsonh = rh.json()
                    if jsonh['name'] != "unknown":
                        planets.append(jsonh['name'])

        url = json["next"]

        if (url is None):
            break
        r = requests.get(url)
        json = r.json()

    return planets
