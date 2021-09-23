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
    while url is not None:
        data = requests.get(url).json()
        for species in data['results']:
            if ((species['designation'] == 'sentient'
                 or species['designation'] == 'reptilian')):
                if species['homeworld'] is not None:
                    jsonh = requests.get(species['homeworld']).json()
                    planets.append(jsonh['name'])
        url = data['next']
    return planets
