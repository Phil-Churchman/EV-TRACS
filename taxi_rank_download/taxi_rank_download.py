import requests
import json

API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"  # Replace with your actual API key
QUERY = "taxi rank in Accra Ghana"
OUTPUT_GEOJSON = "taxi_ranks.geojson"

def get_places(query, api_key):
    url = "https://maps.googleapis.com/maps/api/place/textsearch/json"
    params = {"query": query, "key": api_key}

    results = []
    while True:
        res = requests.get(url, params=params).json()

        for place in res.get("results", []):
            results.append({
                "name": place.get("name"),
                "address": place.get("formatted_address"),
                "lat": place["geometry"]["location"]["lat"],
                "lng": place["geometry"]["location"]["lng"]
            })

        next_token = res.get("next_page_token")
        if next_token:
            import time
            time.sleep(2)
            params = {"pagetoken": next_token, "key": api_key}
        else:
            break

    return results

def to_geojson(features, output_file):
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    for f in features:
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [f["lng"], f["lat"]]
            },
            "properties": {
                "name": f["name"],
                "address": f["address"]
            }
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

# Run
places = get_places(QUERY, API_KEY)
to_geojson(places, OUTPUT_GEOJSON)

print(f"Saved {len(places)} taxi ranks to {OUTPUT_GEOJSON}")
