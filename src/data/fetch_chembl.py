import requests
import pandas as pd
import os

def fetch_activities(target_chembl_id="CHEMBL247", standard_type="IC50", limit=1000):
    """
    Fetches bioactivity data from the ChEMBL API for a given target and activity type.
    Returns a pandas DataFrame.
    """
    base_url = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
    all_data = []
    offset = 0

    while True:
        print(f"Fetching records {offset} to {offset + limit}...")
        params = {
            "target_chembl_id": target_chembl_id,
            "standard_type": standard_type,
            "limit": limit,
            "offset": offset
        }

        response = requests.get(base_url, params=params)
        data = response.json()
        activities = data.get("activities", [])

        if not activities:
            break

        all_data.extend(activities)
        offset += limit

    return pd.DataFrame(all_data)


def save_to_csv(df, filename="data/raw/hiv_ic50_raw.csv"):
    """
    Saves the DataFrame to a CSV file. Creates the directory if it doesn't exist.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"Saved {len(df)} rows to {filename}")

if __name__ == "__main__":
    df = fetch_activities()
    save_to_csv(df)
