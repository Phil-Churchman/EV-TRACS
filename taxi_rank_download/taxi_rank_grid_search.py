import requests
import time
import json

# --- CONFIG ---
API_KEY = "YOUR_GOOGLE_MAPS_API_KEY"  # Replace with your actual API key
CITY_NAME = " Greater Accra"          # Change this to your target city
KEYWORDS = ["taxi rank", "taxi stand", "matatu stage"]
RADIUS = 1000                  # meters for each search
STEP = 0.01                     # ~1.1 km grid spacing
OUTPUT_GEOJSON = "taxi_ranks_GA.geojson"

# --- FUNCTIONS ---
def get_city_bounds(city_name, api_key):
    """Get bounding box (southwest/northeast) of the city from Google Geocoding API"""
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": city_name, "key": api_key}
    response = requests.get(url, params=params).json()

    if not response.get("results"):
        raise ValueError("City not found")

    bounds = response["results"][0]["geometry"]["bounds"] \
        if "bounds" in response["results"][0]["geometry"] \
        else response["results"][0]["geometry"]["viewport"]

    return {
        "min_lat": bounds["southwest"]["lat"],
        "max_lat": bounds["northeast"]["lat"],
        "min_lng": bounds["southwest"]["lng"],
        "max_lng": bounds["northeast"]["lng"]
    }

def search_nearby(lat, lng, keyword, api_key):
    """Search nearby places for a single grid point and keyword"""
    url = "https://maps.googleapis.com/maps/api/place/nearbysearch/json"
    params = {
        "location": f"{lat},{lng}",
        "radius": RADIUS,
        "keyword": keyword,
        "key": api_key
    }

    results = []
    while True:
        response = requests.get(url, params=params).json()

        for place in response.get("results", []):
            results.append(place)

        next_token = response.get("next_page_token")
        if next_token:
            time.sleep(2)
            params = {"pagetoken": next_token, "key": api_key}
        else:
            break

    return results

def run_grid_search(bounds, keywords, api_key):
    """Perform grid-based search across the city bounds"""
    found = {}
    lat = bounds["min_lat"]

    while lat <= bounds["max_lat"]:
        lng = bounds["min_lng"]
        while lng <= bounds["max_lng"]:
            print(f"Searching around {lat:.5f}, {lng:.5f} ...")

            for keyword in keywords:
                results = search_nearby(lat, lng, keyword, api_key)

                for place in results:
                    pid = place.get("place_id")
                    if pid not in found:
                        found[pid] = {
                            "name": place.get("name"),
                            "address": place.get("vicinity") or place.get("formatted_address"),
                            "lat": place["geometry"]["location"]["lat"],
                            "lng": place["geometry"]["location"]["lng"]
                        }

            lng += STEP
        lat += STEP

    return list(found.values())

def to_geojson(data, output_file):
    """Convert results to GeoJSON"""
    geojson = {
        "type": "FeatureCollection",
        "features": []
    }

    for place in data:
        geojson["features"].append({
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [place["lng"], place["lat"]]
            },
            "properties": {
                "name": place["name"],
                "address": place["address"]
            }
        })

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)

# --- MAIN ---
if __name__ == "__main__":
    print(f"Fetching bounding box for {CITY_NAME}...")
    city_bounds = get_city_bounds(CITY_NAME, API_KEY)
    print(f"City bounds: {city_bounds}")

    all_places = run_grid_search(city_bounds, KEYWORDS, API_KEY)
    print(f"Total unique taxi ranks found: {len(all_places)}")

    to_geojson(all_places, OUTPUT_GEOJSON)
    print(f"GeoJSON saved to {OUTPUT_GEOJSON}")
