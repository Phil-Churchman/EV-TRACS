import os
import json
import math
import random
import multiprocessing
import heapq
import functools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
import osmnx as ox
import networkx as nx
from tqdm import tqdm
from scipy.spatial import KDTree
from shapely.geometry import Point
from trip_demand_generator import generate_trips
from fill_gaps import insert_idle_points    

# ============================================================
# CONFIGURATION & GLOBALS
# ============================================================
PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
try:
    with open(os.path.join(PARENT_DIR, "scenario.json"), "r") as f:
        scenario_cfg = json.load(f)
except FileNotFoundError:
    with open("scenario.json", "r") as f:
        scenario_cfg = json.load(f)

DEMAND_MODEL = scenario_cfg.get("demand_model", False)
FOLDER_NAME = scenario_cfg["folder_name"]
ROAD_SPEEDS = scenario_cfg["road_speed_km-h"]
ROAD_SPEEDS_MS = {k: v / 3.6 for k, v in ROAD_SPEEDS.items()}
NUM_AGENTS_CFG = scenario_cfg["agents"] 
INPUT_DIR = os.path.join(PARENT_DIR, FOLDER_NAME, "geojson_files")
ROAD_NETWORK_FILE = os.path.join(INPUT_DIR, "roads.graphml")
SWAP_STATIONS_FILE = os.path.join(INPUT_DIR, "swap_stations.geojson")
TAXI_RANKS_FILE = os.path.join(INPUT_DIR, "taxi_ranks.geojson")

OUTPUT_DIR = os.path.join(PARENT_DIR, FOLDER_NAME, "output")
OUTPUT_PER_AGENT_DIR = os.path.join(OUTPUT_DIR, "output_trips")
OUTPUT_PER_AGENT_TIME_DIR = os.path.join(OUTPUT_DIR, "output_trips_time_queued")
HISTOGRAM_PLOT = os.path.join(OUTPUT_DIR, "station_queues_analysis.png")

# --- SIMULATION PARAMETERS ---
SIMULATION_INTERVAL_SEC = scenario_cfg.get("simulation_step_sec", 60) 
SWAP_WAIT_SEC = scenario_cfg.get("swap_wait_sec", 600)
d_start, d_end = scenario_cfg["start_time"], scenario_cfg["end_time"]
DAY_START = datetime(d_start[0], d_start[1], d_start[2], d_start[3], d_start[4], d_start[5])
DAY_END = datetime(d_end[0], d_end[1], d_end[2], d_end[3], d_end[4], d_end[5])
MAX_TOTAL_DISTANCE_M = scenario_cfg.get("max_total_distance_m", 80000)
BUFFER_DISTANCE = scenario_cfg.get("buffer_distance", 12000)
PASSENGER_MAX_DIST = scenario_cfg.get("passenger_max_dist", 6000)
PROBABILITY_OF_HAILING_TAXI = scenario_cfg.get("probability_hail", 0.75)
NUM_NEAREST_TAXI_RANKS_FIRST = 4

# --- ROUTING LOGIC ---
SPEED_BASED_ROUTING = scenario_cfg.get("speed_based_routing", False)
ROUTING_WEIGHT = "travel_time" if SPEED_BASED_ROUTING else "length"

# Workers Globals
G_GLOBAL = None
TREES_GLOBAL = None
NODES_GLOBAL = None
SWAP_NODES_GLOBAL = None
TAXI_NODES_GLOBAL = None
COORD_LOOKUP = None
EDGE_LOOKUP = None

unallocated_demand = []
allocated_demand = []
met_demand = []


# ============================================================
# CLASSES
# ============================================================

class SwapStation:
    def __init__(self, station_id, num_servers=2):
        self.station_id = station_id
        self.num_servers = num_servers
        self.servers_free_at = [0] * num_servers
        heapq.heapify(self.servers_free_at)
        self.active_taxis = []
        self.queue_history = []

    def process_arrival(self, arrival_sec, duration_sec):
        earliest_free = self.servers_free_at[0]
        start_sec = max(arrival_sec, earliest_free)
        finish_sec = start_sec + duration_sec
        heapq.heapreplace(self.servers_free_at, finish_sec)
        heapq.heappush(self.active_taxis, (finish_sec, arrival_sec))
        return finish_sec

    def record_queue(self, current_sec):
        while self.active_taxis and self.active_taxis[0][0] <= current_sec:
            heapq.heappop(self.active_taxis)
        self.queue_history.append(len(self.active_taxis))

class TaxiAgent:
    def __init__(self, agent_id, start_node, spawn_time):
        self.id = agent_id
        self.pos = start_node
        self.spawn_time = spawn_time
        self.running_total = random.uniform(5000, 60000)
        self.trip_count = 0
        self.busy_until = 0
        self.state = "IDLE" 
        self.current_trip_type = None # Track the type of the active trip
        self.target_node = None
        self.pending_wait_sec = 0
        self.agent_features = []
        self.time_features = []
        self.marked_for_removal = False
        self.allocated_trip = None
        self.assigned_trip = False

# ============================================================
# WORKER INITIALIZATION & CACHING
# ============================================================

def init_worker(g, trees, nodes, swap_nodes, taxi_nodes, coords, edge_lookup):
    global G_GLOBAL, TREES_GLOBAL, NODES_GLOBAL, SWAP_NODES_GLOBAL, TAXI_NODES_GLOBAL, COORD_LOOKUP, EDGE_LOOKUP
    G_GLOBAL, TREES_GLOBAL, NODES_GLOBAL, SWAP_NODES_GLOBAL, TAXI_NODES_GLOBAL, COORD_LOOKUP, EDGE_LOOKUP = g, trees, nodes, swap_nodes, taxi_nodes, coords, edge_lookup

@functools.lru_cache(maxsize=30000)
def get_cached_path(u, v):
    return nx.shortest_path(G_GLOBAL, u, v, weight=ROUTING_WEIGHT)

@functools.lru_cache(maxsize=30000)
def get_cached_cost(u, v):
    return nx.shortest_path_length(G_GLOBAL, u, v, weight=ROUTING_WEIGHT)

@functools.lru_cache(maxsize=10000)
def get_physical_distance(u, v):
    return nx.shortest_path_length(G_GLOBAL, u, v, weight="length")

# ============================================================
# HELPERS
# ============================================================

def to_serializable(obj):
    if isinstance(obj, (np.integer, np.int64)): return int(obj)
    if isinstance(obj, (np.floating, np.float64)): return float(obj)
    return obj

def get_target_node(agent_pos, trip_type):
    tree_nodes, tree_swap, tree_taxi = TREES_GLOBAL
    agent_xy = COORD_LOOKUP[agent_pos]
    
    if trip_type == "to_swap":
        _, idxs = tree_swap.query(agent_xy, k=min(3, len(SWAP_NODES_GLOBAL)))
        candidates = [SWAP_NODES_GLOBAL[i] for i in (idxs if hasattr(idxs, "__len__") else [idxs])]
        best_node, min_cost = candidates[0], float('inf')
        for cand in candidates:
            try:
                cost = get_cached_cost(agent_pos, cand)
                if cost < min_cost: min_cost, best_node = cost, cand
            except: continue
        return best_node

    elif trip_type in ["taxi", "hail"]:
        _, idxs = tree_taxi.query(agent_xy, k=min(NUM_NEAREST_TAXI_RANKS_FIRST, len(TAXI_NODES_GLOBAL)))
        candidates = [TAXI_NODES_GLOBAL[i] for i in (idxs if hasattr(idxs, "__len__") else [idxs])]
        costs = []
        for cand in candidates:
            try:
                costs.append((get_cached_cost(agent_pos, cand), cand))
            except: continue
        costs.sort()
        return costs[0][1] if costs else random.choice(TAXI_NODES_GLOBAL)

    elif trip_type == "passenger":
        possible_indices = tree_nodes.query_ball_point(agent_xy, PASSENGER_MAX_DIST)
        if possible_indices:
            random.shuffle(possible_indices)
            for idx in possible_indices[:5]:
                target = NODES_GLOBAL[idx]
                try:
                    if get_physical_distance(agent_pos, target) <= PASSENGER_MAX_DIST:
                        return target, get_cached_path(agent_pos, target)
                except: continue
        t = random.choice(NODES_GLOBAL)
        try:
            return t, get_cached_path(agent_pos, t)
        except:
            return t, [agent_pos, t]

def calculate_next_activity(args):
    agent, current_sec = args
    is_swap = (agent.running_total + BUFFER_DISTANCE > MAX_TOTAL_DISTANCE_M and agent.trip_count % 2 == 0)

    if not DEMAND_MODEL:

        if is_swap:
            trip_type = "to_swap"
        elif agent.trip_count % 2 == 0:
            trip_type = "hail" if random.random() < PROBABILITY_OF_HAILING_TAXI else "taxi"
        else:
            trip_type = "passenger"

    else:
        if is_swap:
            trip_type = "to_swap"
            agent.assigned_trip = False
        elif agent.assigned_trip and agent.allocated_trip:
            if agent.trip_count % 2 == 0:
                trip_type = "pickup"
            else:
                trip_type = "passenger"

                agent.assigned_trip = False # Mark trip as now being served
        else:
            return agent
            

    agent.current_trip_type = trip_type # FIX: Persist type for the wait segment later

    wait_sec_raw = SWAP_WAIT_SEC if trip_type == "to_swap" else (60 if trip_type == "hail" else (random.randint(1, 20) * 60 if trip_type == "taxi" else 60))
    wait_sec = math.ceil(wait_sec_raw / SIMULATION_INTERVAL_SEC) * SIMULATION_INTERVAL_SEC

    if trip_type == "passenger":
        if DEMAND_MODEL and agent.allocated_trip:
            demand = agent.allocated_trip
            target_node, route = demand["dest_node"], get_cached_path(agent.pos, demand["dest_node"])
        else:
            target_node, route = get_target_node(agent.pos, trip_type)
    elif trip_type == "pickup":
        demand = agent.allocated_trip
        target_node = demand["source_node"]
        try:
            route = get_cached_path(agent.pos, target_node)
        except:
            route = [agent.pos, target_node]
    else:
        target_node = get_target_node(agent.pos, trip_type)
        try:
            route = get_cached_path(agent.pos, target_node)
            if trip_type == "hail" and len(route) > 2:
                route = route[:random.randint(2, len(route))]
                target_node = route[-1]
        except: route = [agent.pos, target_node]

    total_len, total_time, segments, coords = 0, 0, [], [COORD_LOOKUP[route[0]]]
    for u, v in zip(route[:-1], route[1:]):
        edge = EDGE_LOOKUP.get((u, v))
        if edge:
            segments.append(edge); total_len += edge["length_m"]; total_time += edge["travel_time_s"]
        coords.append(COORD_LOOKUP[v])

    travel_sec = max(SIMULATION_INTERVAL_SEC, math.ceil(total_time / SIMULATION_INTERVAL_SEC) * SIMULATION_INTERVAL_SEC)
    geom = {"type": "LineString", "coordinates": coords} if (len(coords) > 1 and coords[0] != coords[-1]) else {"type": "Point", "coordinates": coords[0]}

    start_dt = DAY_START + timedelta(seconds=current_sec)
    end_dt = start_dt + timedelta(seconds=travel_sec)

    feat = {
        "type": "Feature", "geometry": geom,
        "properties": {
            "agent": agent.id, "type": trip_type, "length_m": total_len,
            "total_distance_m": agent.running_total, "start_time": start_dt.isoformat(),
            "end_time": end_dt.isoformat(), "duration_s": travel_sec, "segments": segments
        }
    }
    agent.time_features.append(feat)
    agent.state = "DRIVING_TO_SWAP" if trip_type == "to_swap" else "DRIVING"
    agent.busy_until, agent.pending_wait_sec, agent.pos, agent.target_node = current_sec + travel_sec, wait_sec, target_node, target_node

    if trip_type == "to_swap": agent.running_total, agent.trip_count = 0, 0
    else: agent.running_total += total_len; agent.trip_count += 1
    if DEMAND_MODEL and trip_type == "passenger":
        request_time = agent.allocated_trip["departure_time"]
        pickup_time = (DAY_START + timedelta(seconds=current_sec)).isoformat()

        met_demand.append({"idx": agent.allocated_trip["id"], "request_time": request_time, "pickup_time": pickup_time, "source_facility": agent.allocated_trip["source_facility"], "dest_facility": agent.allocated_trip["dest_facility"], "source_category": agent.allocated_trip["source_category"], "dest_category": agent.allocated_trip["dest_category"]})

    return agent

# ============================================================
# MAIN
# ============================================================

def main():
    G = ox.load_graphml(ROAD_NETWORK_FILE)
    print(f"Routing Mode: {'TIME' if SPEED_BASED_ROUTING else 'DISTANCE'} (Weight: {ROUTING_WEIGHT})")
    
    edge_lookup = {}
    for u, v, k, data in G.edges(keys=True, data=True):
        hw = data.get("highway")
        speed = max([ROAD_SPEEDS_MS.get(h, 0.1) for h in hw]) if isinstance(hw, list) else ROAD_SPEEDS_MS.get(hw, 0.1)
        length = data.get("length", 1.0)
        travel_time = length / speed
        if (u, v) not in edge_lookup or travel_time < edge_lookup[(u, v)]["travel_time_s"]:
            edge_lookup[(u, v)] = {"u": int(u), "v": int(v), "highway": hw, "speed_m_s": float(speed), "length_m": float(length), "travel_time_s": float(travel_time)}

    nodes = list(G.nodes())
    coords_dict = {n: (float(G.nodes[n]['x']), float(G.nodes[n]['y'])) for n in nodes}

    def load_pts(path):
        with open(path) as f: data = json.load(f)
        pts = [Point(feat["geometry"]["coordinates"]) for feat in data["features"]]
        return list(ox.nearest_nodes(G, [p.x for p in pts], [p.y for p in pts]))

    # if demand model, load demand

    if DEMAND_MODEL:
        print("Generating trip demand using the demand model...")
        trips=[]
        features = generate_trips()["features"]
        idx = 0
        for feat in features:
            trips.append({
                "id": idx,
                "departure_time": feat["properties"]["departure_time"],
                "source_facility": feat["properties"]["source_facility"],
                "dest_facility": feat["properties"]["dest_facility"],
                "source_category": feat["properties"]["source_category"],
                "dest_category": feat["properties"]["dest_category"],
                "source_node": ox.nearest_nodes(G, feat["geometry"]["coordinates"][0][0], feat["geometry"]["coordinates"][0][1]),
                "dest_node": ox.nearest_nodes(G, feat["geometry"]["coordinates"][-1][0], feat["geometry"]["coordinates"][-1][1])
          })
            idx += 1
        trips.sort(key=lambda x: x["departure_time"])

    swap_nodes = load_pts(SWAP_STATIONS_FILE)
    if not DEMAND_MODEL:
        taxi_nodes = load_pts(TAXI_RANKS_FILE)
        trees = (KDTree([coords_dict[n] for n in nodes]), KDTree([coords_dict[n] for n in swap_nodes]), KDTree([coords_dict[n] for n in taxi_nodes]))
    else:
        taxi_nodes = []
        trees = (KDTree([coords_dict[n] for n in nodes]), KDTree([coords_dict[n] for n in swap_nodes]), None)

    with open(SWAP_STATIONS_FILE) as f: station_data = json.load(f)
    stations = {swap_nodes[i]: SwapStation(feat["properties"].get("facility_id", i), int(feat["properties"].get("posts", 2))) for i, feat in enumerate(station_data["features"])}

    total_seconds = int((DAY_END - DAY_START).total_seconds())
    agent_counts = NUM_AGENTS_CFG if isinstance(NUM_AGENTS_CFG, list) else [NUM_AGENTS_CFG]
    num_periods, period_duration = len(agent_counts), total_seconds / len(agent_counts)

    agents, retired_agents, next_agent_id, current_period_idx = [], [], 0, -1
    pool = multiprocessing.Pool(initializer=init_worker, initargs=(G, trees, nodes, swap_nodes, taxi_nodes, coords_dict, edge_lookup))

    for s in tqdm(range(0, total_seconds, SIMULATION_INTERVAL_SEC), desc="Simulating"):
        new_period_idx = min(int(s // period_duration), num_periods - 1)

        if DEMAND_MODEL:
            while True:
                if trips and trips[0]["departure_time"] <= (DAY_START + timedelta(seconds=s)).isoformat():
                    unallocated_demand.append(trips.pop(0))
                else:
                    break

            ## Assign closest idle agents to unallocated demand ##    
            idle_idxs = [i for i, a in enumerate(agents) if a.state == "IDLE" and not a.assigned_trip]
            while unallocated_demand and idle_idxs:
                
                min_distance = float('inf')
                for idx in idle_idxs:
                    agent = agents[idx]
                    try:
                        dist = get_physical_distance(agent.pos, unallocated_demand[0]["source_node"])
                        if dist < min_distance:
                            min_distance = dist
                            best_agent_idx = idx
                    except: continue
                agents[best_agent_idx].allocated_trip = unallocated_demand.pop(0)
                agents[best_agent_idx].assigned_trip = True
                idle_idxs.remove(best_agent_idx)

        if new_period_idx > current_period_idx:
            target = agent_counts[new_period_idx]
            active = [a for a in agents if not a.marked_for_removal]
            if target > len(active):
                for _ in range(target - len(active)):
                    if not DEMAND_MODEL:
                        initial_node = random.choice(nodes)
                    else:
                        initial_node = random.choice(swap_nodes)
                    agents.append(TaxiAgent(next_agent_id, initial_node, spawn_time=s)); next_agent_id += 1

            elif target < len(active):
                active.sort(key=lambda x: x.spawn_time)
                for a in active[:(len(active)-target)]: a.marked_for_removal = True
            current_period_idx = new_period_idx

        still_active = []
        for a in agents:
            if s >= a.busy_until:
                if a.state in ["DRIVING", "DRIVING_TO_SWAP"]:
                    if a.state == "DRIVING_TO_SWAP":
                        station = stations[a.target_node]
                        finish_s = station.process_arrival(s, a.pending_wait_sec)
                        wait_type = "to_swap"
                        extra_props = {"facility_id": station.station_id}
                    else:
                        finish_s = s + a.pending_wait_sec
                        wait_type = a.current_trip_type # FIX: Use stored trip type
                        extra_props = {}

                    feat = {
                        "type": "Feature", "geometry": {"type": "Point", "coordinates": coords_dict[a.pos]},
                        "properties": {
                            "agent": a.id, "type": wait_type, "duration_s": (finish_s - s),
                            "start_time": (DAY_START + timedelta(seconds=s)).isoformat(),
                            "end_time": (DAY_START + timedelta(seconds=finish_s)).isoformat(),
                            **extra_props
                        }
                    }
                    a.agent_features.append(feat); a.time_features.append(feat)
                    a.busy_until, a.state = finish_s, "WAITING_COMPLETE"
                elif a.marked_for_removal: retired_agents.append(a); continue
                else: a.state = "IDLE"
            still_active.append(a)
        agents = still_active
        for stat in stations.values(): stat.record_queue(s)

        idle_idxs = [i for i, a in enumerate(agents) if a.state == "IDLE"]
        if len(idle_idxs) > 20:
            results = pool.map(calculate_next_activity, [(agents[i], s) for i in idle_idxs])
            for i, updated in zip(idle_idxs, results): agents[i] = updated
        elif idle_idxs:
            if EDGE_LOOKUP is None: init_worker(G, trees, nodes, swap_nodes, taxi_nodes, coords_dict, edge_lookup)
            for i in idle_idxs: agents[i] = calculate_next_activity((agents[i], s))

    pool.close(); retired_agents.extend(agents)
    plot_station_queues(stations, HISTOGRAM_PLOT)

    with open(os.path.join(OUTPUT_DIR, "met_demand.json"), "w") as f: json.dump(met_demand, f, default=to_serializable)

    unmet_demand = [{"idx": d["id"], "request_time": d["departure_time"], "source_facility": d["source_facility"], "dest_facility": d["dest_facility"], "source_category": d["source_category"], "dest_category": d["dest_category"]} for d in unallocated_demand]
    with open(os.path.join(OUTPUT_DIR, "unmet_demand.json"), "w") as f: json.dump(unmet_demand, f, default=to_serializable)

    os.makedirs(OUTPUT_DIR, exist_ok=True); os.makedirs(OUTPUT_PER_AGENT_DIR, exist_ok=True); os.makedirs(OUTPUT_PER_AGENT_TIME_DIR, exist_ok=True)
    for a in tqdm(retired_agents, desc="Saving"):
        time_geojson = insert_idle_points({"type":"FeatureCollection","features":a.time_features})
        with open(f"{OUTPUT_PER_AGENT_DIR}/agent_{a.id:04d}.geojson","w") as f: json.dump({"type":"FeatureCollection","features":a.agent_features}, f, default=to_serializable)
        with open(f"{OUTPUT_PER_AGENT_TIME_DIR}/agent_{a.id:04d}_time.geojson","w") as f: json.dump(time_geojson, f, default=to_serializable)

def plot_station_queues(stations, output_path):
    active_stations = sorted([s for s in stations.values() if s.queue_history], key=lambda x: x.station_id)
    json.dump({s.station_id: s.queue_history for s in stations.values() if s.queue_history}, open(os.path.join(OUTPUT_DIR, "queue_history.json"), "w"), default=to_serializable)
    if not active_stations: return
    n_stations = len(active_stations)
    cols = 1 if n_stations == 1 else 2
    rows = (n_stations + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(12 if cols == 1 else 24, rows * 5), sharex=True, sharey=True, squeeze=False)
    flat_axes = axes.flatten()
    
    # Calculate max pressure for consistent y-axis scaling
    global_y_max = max(max(s.queue_history) / s.num_servers for s in active_stations)
    
    for i, s in enumerate(active_stations):
        ax = flat_axes[i]
        times = [DAY_START + timedelta(seconds=sec) for sec in range(0, len(s.queue_history)*SIMULATION_INTERVAL_SEC, SIMULATION_INTERVAL_SEC)]
        pressure = np.array(s.queue_history) / s.num_servers
        
        ax.step(times, pressure, where='post', color='#2c3e50', lw=2.5)
        ax.fill_between(times, pressure, step='post', alpha=0.3, color='#3498db')
        ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.8, lw=2.5)
        
        # Formatting
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, pos: mdates.num2date(x).strftime('%I%p').lower().lstrip('0')))
        ax.set_ylim(0, global_y_max * 1.1)
        
        # --- MODIFIED LINE BELOW ---
        ax.set_title(f"Station {s.station_id} ({s.num_servers} Posts)", fontsize=18, fontweight='bold')
        # ---------------------------

    # Hide unused subplots if any
    for j in range(i + 1, len(flat_axes)):
        flat_axes[j].axis('off')

    plt.tight_layout(pad=4.0)
    plt.savefig(output_path, dpi=300)

if __name__ == "__main__":
    main()