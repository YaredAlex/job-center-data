# pipeline: map positions -> sector & subsector

import json
import pandas as pd
import requests
import time
from dotenv import load_dotenv
import os
load_dotenv(override=True)


API_KEY = os.getenv("DEEPSEEK_API_KEY",None)
API_URL = "https://api.deepseek.com/v1/chat/completions"


def classifier(positions, sector_sub_sectors):
    """
    Send batch of positions to DeepSeek and return parsed JSON result
    """

    prompt = f"""
You are a classifier that maps job positions to sector and subsector.

Here is the sector and subsector JSON:
{json.dumps(sector_sub_sectors, indent=2)}

Rules:
- Choose ONLY ONE best match
- If unsure, return "None"
- Response MUST be valid JSON only
- No explanation text
- Strictly return json value containing position, sector and sub_sector keys
Example:
{
  {
   "position":"Branch Manager",
    "sector": "Business",
    "sub_sector": "Wholesale business in specialized stores"
  }
}
"""

    try:
        response = requests.post(
            API_URL,
            headers={
                "Authorization": f"Bearer {API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek-chat",
                "messages": [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": json.dumps(positions)}
                ],
                "temperature": 0
            }
        )

        result = response.json()

        content = result["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(content)
        except Exception as e:
            print("Error parsing response content ", e)
            start = content.find("{")
            end = content.rfind("}") + 1
            parsed = json.loads(content[start:end])

        return parsed

    except Exception as e:
        print("Error:", e)
        return {}


def get_unmapped_positions(df):
    """
    Filter only rows that need mapping
    """
    # Ensure columns exist
    if "sector" not in df.columns:
        df["sector"] = None
    if "sub sector" not in df.columns:
        df["sub sector"] = None

    # Unmapped = sector or sub sector is missing/empty
    mask = (
        df["sector"].isna() | (df["sector"] == "") |
        df["sub sector"].isna() | (df["sub sector"] == "")
    )

    unmapped_df = df[mask]

    # Remove duplicates to reduce token usage
    positions = unmapped_df["informal work in eng"].dropna().unique().tolist()

    return positions, mask

def apply_results_to_df(df, results):
    """
    Update dataframe with sector & subsector
    Handles new result format (list of dicts)
    """

    print("=" * 50)
    print("results are:", results)
    print("=" * 50)

    # ✅ Normalize results into dict: {position: {...}}
    result_map = {}

    if isinstance(results, list):
        for item in results:
            pos = item.get("position")
            if pos:
                result_map[pos] = item

    elif isinstance(results, dict):
        # fallback if model sometimes returns old format
        result_map = results

    # ✅ Update dataframe
    for idx, row in df.iterrows():
        position = row["informal work in eng"]

        if position in result_map:
            df.at[idx, "sector"] = result_map[position].get("sector")
            df.at[idx, "sub sector"] = result_map[position].get("sub_sector")

    return df


def batch_process(positions, sector_json, df, batch_size=20, output_file="output_with_sectors.csv"):
    """
    Production batching with incremental saving after each batch
    """

    all_results = {}

    total_batches = (len(positions) // batch_size) + 1

    for i in range(0, len(positions), batch_size):
        batch_num = i // batch_size + 1
        batch = positions[i:i + batch_size]

        print(f"Processing batch {batch_num} / {total_batches}...")

        result = classifier(batch, sector_json)

        # accumulate
        # all_results.update(result)

        # ✅ apply only this batch result
        df = apply_results_to_df(df, result)

        # ✅ save progress after each batch
        df.to_csv(output_file, index=False)
        print(f"✅ Saved progress after batch {batch_num}")

        # avoid rate limit
        time.sleep(1)

    return all_results, df


def sector_subsector_mapper(test_mode=True):
    # load sector-subsector
    with open("sub_sectors.json", "r") as file:
        sub_sectors_json = json.load(file)

    # load positions
    df_positions = pd.read_csv("job positions - Tibarek Final.csv")
    if os.path.exists("output_with_sectors.csv"):
        df_positions = pd.read_csv("output_with_sectors.csv")
    #positions = df_positions['informal work in eng'].dropna().tolist()
    positions, mask = get_unmapped_positions(df_positions)

    if test_mode:
        print("Running TEST MODE (first 10 positions)")
        results = classifier(positions[:10], sub_sectors_json)
    else:
        print("Running PRODUCTION MODE (batch processing)")
        # results = batch_process(positions, sub_sectors_json, batch_size=25)
        _, df_positions = batch_process(
                                        positions,
                                        sub_sectors_json,
                                        df_positions,
                                        batch_size=10,
                                        output_file="output_with_sectors.csv")
                                                                                
    if test_mode:
        # apply results
        df_positions = apply_results_to_df(df_positions, results)
        # save output
        df_positions.to_csv("output_with_sectors.csv", index=False)

    print("✅ Processing completed. Output saved to output_with_sectors.csv")


if __name__ == "__main__":
    # change to False for production
    sector_subsector_mapper(test_mode=False)