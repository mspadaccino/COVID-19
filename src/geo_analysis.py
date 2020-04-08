import json
import urllib.request as url_req
import time
import pandas as pd
import populartimes
import tqdm
with open ('api_key.cfg') as file:
    API_KEY = file.readline().strip()

with open('../data/geodata/places_types.config') as f:
    places_types = f.readlines()
    places_types = [item.replace('\n','') for item in places_types]


MILAN_CENTER = (45.4654219, 9.1859243) # Milan Coordinates
bounding_box_milan_lower = [45.390368, 9.092639]
bounding_box_milan_upper = [45.539202,9.284036]

API_NEARBY_SEARCH_URL = 'https://maps.googleapis.com/maps/api/place/nearbysearch/json'
RADIUS = 30000

PLACES_TYPES = [('airport', 1), ('bank', 2), ('bar', 3), ('beauty_salon', 3), ('book_store', 1), ('cafe', 1), ('church', 3), ('doctor', 3), ('dentist', 2), ('gym', 3), ('hair_care', 3), ('hospital', 2), ('pharmacy', 3), ('pet_store', 1), ('night_club', 2), ('movie_theater', 1), ('school', 3), ('shopping_mall', 1), ('supermarket', 3), ('store', 3)]

def request_api(url):
    response = url_req.urlopen(url)
    json_raw = response.read()
    json_data = json.loads(json_raw)
    return json_data

def get_places(types, pages):
    location = str(MILAN_CENTER[0]) + "," + str(MILAN_CENTER[1])
    mounted_url = ('%s'
                   '?location=%s'
                   '&radius=%s'
                   '&type=%s'
                   '&key=%s') % (API_NEARBY_SEARCH_URL, location, RADIUS, types, API_KEY)

    results = []
    next_page_token = None

    if pages == None: pages = 1

    for num_page in range(pages):
        if num_page == 0:
            api_response = request_api(mounted_url)
            results = results + api_response['results']
        else:
            page_url = ('%s'
                        '?key=%s'
                        '&pagetoken=%s') % (API_NEARBY_SEARCH_URL, API_KEY, next_page_token)
            api_response = request_api(str(page_url))
            results += api_response['results']

        if 'next_page_token' in api_response:
            next_page_token = api_response['next_page_token']
        else:
            break

        time.sleep(1)
    return results


def parse_place_to_list(place, type_name):
    # Using name, place_id, lat, lng, rating, type_name
    return [
        place['name'],
        place['place_id'],
        place['geometry']['location']['lat'],
        place['geometry']['location']['lng'],
        type_name
    ]


def update_places_dataset():
    data = []
    for place_type in PLACES_TYPES:
        type_name = place_type[0]
        type_pages = place_type[1]
        print("Getting into " + type_name)
        result = get_places(type_name, type_pages)
        result_parsed = list(map(lambda x: parse_place_to_list(x, type_name), result))
        data += result_parsed

    dataframe = pd.DataFrame(data, columns=['place_name', 'place_id', 'lat', 'lng', 'type'])
    dataframe.to_csv('../data/places.csv')
    return dataframe


def get_place_details(place_id):
    poptimes = populartimes.get_id(API_KEY, place_id)
    return pd.json_normalize(poptimes)


def get_places_popularity(places):
    places_popularity = pd.DataFrame()
    for row, item in tqdm.tqdm(places.iterrows()):
        places_popularity = places_popularity.append(get_place_details(item.place_id), ignore_index=True)
    places_popularity.to_csv('places_popularity.csv')
    return places_popularity


if __name__ == '__main__':

    # test = populartimes.get(API_KEY, ['bank', 'hotel'], bounding_box_milan_lower, bounding_box_milan_upper, 1,radius=100)

    UPDATE_PLACES_DATASET = False
    if UPDATE_PLACES_DATASET:
        places = update_places_dataset()
    else:
        places = pd.read_csv('../data/places.csv', index_col=0)

    GET_POPULARITY = False
    if GET_POPULARITY:
        places_popularity = get_places_popularity(places)
    else:
        places_popularity = pd.read_csv('../data/places_popularity.csv', index_col=0)

        # places = pd.read_csv('../data/places.csv', index_col=0)
        # places['monday'] = None
        # places['tuesday'] = None
        # places['wednesday'] = None
        # places['thursday'] = None
        # places['friday'] = None
        # places['saturday'] = None
        # places['sunday'] = None
        # places['current'] = None
        # for (index, row) in places.iterrows():
        #     print("Populating " + str(index))
        #     moments = get_place_popular_moments(row.place_id)
        #     if moments != None:
        #         places.at[index, 'monday'] = moments[0]['data']
        #         places.at[index, 'tuesday'] = moments[1]['data']
        #         places.at[index, 'wednesday'] = moments[2]['data']
        #         places.at[index, 'thursday'] = moments[3]['data']
        #         places.at[index, 'friday'] = moments[4]['data']
        #         places.at[index, 'saturday'] = moments[5]['data']
        #         places.at[index, 'sunday'] = moments[6]['data']
        #         places.at[index, 'sunday'] = moments[6]['data']
        #
        # places.to_csv('../data/places_with_moments.csv')
        # print(places)

