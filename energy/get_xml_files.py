import pandas as pd 
from obspy.clients.fdsn import Client
import os


if __name__ == "__main__":
    df = pd.read_csv('../assets/available_fdsn_stations.csv')

    client_iris = Client("IRIS")
    client_geofon = Client("GFZ")

    client = {
        "IRIS": client_iris,
        "GEOFON": client_geofon
    }
    os.makedirs(f"../assets/inventory/", exist_ok=True)
    for idx, row in df.iterrows():
        network = row['network']
        station = row['station']
        datacenter = "GEOFON" if network == "CX" else "IRIS"
        print(f"Downloading inventory for {network}.{station}")

        try:
            inv = client[datacenter].get_stations(network=network, station=station, channel="BHZ", level="response", format="XML")
            inv.write(f"../assets/inventory/{network}.{station}.xml", format="stationxml")
            print(f"Success for inventory for {network}.{station}")
        except Exception as e:
            print(f"Error {network}.{station}")
            print(e)