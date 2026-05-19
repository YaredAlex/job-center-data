# pipeline: normalize and deduplicate job positions

import json
import math
import os
import time

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

API_KEY = os.getenv("DEEPSEEK_API_KEY", None)

API_URL = "https://api.deepseek.com/v1/chat/completions"


def classifier(positions):
    """
    Normalize and deduplicate job positions using DeepSeek
    """

    prompt = """
IMPORTANT:
You are NOT a text formatter.
You are an occupation normalization and validation system.

Your job is to:
- normalize valid occupations
- remove organizational/contextual noise
- reject non-standard occupations
- map invalid occupations to null

A valid occupation MUST:
- represent a recognized professional role, trade, or occupation
- be understandable without organizational context
- exist as a commonly recognized labor-market occupation

INVALID TITLES:
Map to null if the title:
- is only a department/unit/function
- is organization-specific
- is not a recognized occupation
- is vague or ambiguous
- is merely an assignment area or cooperation area
- contains only geographic/political/business context
- is just a heading or responsibility area

Examples:

WRONG:
- "China Cooperation Expert"
→ null

Reason:
"China Cooperation" is an organizational/program area, not a profession.

WRONG:
- "China Cooperation Division Head"
→ null

Reason:
Division/unit heads without a standardized profession should be rejected.

DO NOT simply capitalize or rewrite invalid titles.

--------------------------------------------------

NORMALIZATION RULES

1. Normalize only REAL occupations.

2. Remove company names, department names, geographic references, project names, and organizational units.

Examples:
- "Google Software Engineer"
→ "Software Engineer"

- "China Cooperation Expert"
→ null

3. Remove contextual modifiers that do NOT change the occupation itself.

Examples:
- "Children Face Paint Artist"
→ "Face Paint Artist"

- "Children's Boutique Sales Clerk"
→ "Sales Clerk"

- "Children's Hairdresser"
→ "Hairdresser"

4. KEEP domain-specific words ONLY when they fundamentally define the profession.

KEEP:
- "Child Rights and Protection Executive"
- "Child Support Service Specialist"
- "Child, Family, and School Social Worker"

Reason:
"Child" changes the actual specialization and occupation domain.

5. Do NOT preserve meaningless descriptors.

REMOVE:
- cooperation
- division
- unit
- desk
- section
- program
- initiative
- project
- office

unless they are part of a globally recognized profession.

6. Do NOT invent occupations.

7. Do NOT infer hierarchy or organizational roles.

8. Do NOT generalize valid specialized occupations into overly broad categories.

9. If uncertain whether the title is a real occupation:
→ return null

10. Return ONLY valid JSON.

11. No markdown.

12. No explanations.
IMPORTANT OCCUPATION INFERENCE RULE

If a title is not itself a standard occupation BUT clearly describes:
- a professional function,
- institutional responsibility,
- operational role,
- or administrative responsibility

then map it to the closest recognized professional occupation instead of null.

The inferred occupation MUST:
- already exist as a real labor-market occupation
- preserve the professional domain
- preserve the operational responsibility
- NOT invent organizational hierarchy
- NOT create fictional titles

Examples of valid inference behavior:
- titles related to academic administration, curriculum, scheduling, review, staff affairs, or program coordination
should map to real academic administrative occupations.

- operational responsibilities involving audit, scheduling, coordination, monitoring, or affairs
should map to corresponding officer/coordinator/specialist/administrator roles when clearly implied.

DO NOT return vague department names or activity names alone.

--------------------------------------------------

NULL RULE

Return null ONLY if:
- no recognizable profession exists,
- the title is purely organizational,
- the title is only a business/program/geographic label,
- or the role cannot reasonably map to a recognized occupation.

--------------------------------------------------

DOMAIN PRESERVATION RULE

Preserve meaningful professional domains when normalizing.

Examples of meaningful domains:
- academic
- curriculum
- child protection
- finance
- procurement
- legal
- nursing
- engineering

Do NOT remove domain words that fundamentally define the professional field.

--------------------------------------------------

ROLE EXTRACTION RULE

When a title describes:
- an activity,
- process,
- responsibility,
- or unit

extract the implied occupation from the responsibility.

Examples:
- scheduling and monitoring
→ coordinator / officer role

- staff affairs
→ human resources / staff administration role

- curriculum preparation
→ curriculum specialist role

- academic review
→ academic quality assurance role

Required format:

[
  {
    "position": "raw title",
    "new_position": "normalized occupation or null"
  }
]
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
                    {
                        "role": "system",
                        "content": prompt
                    },
                    {
                        "role": "user",
                        "content": json.dumps(
                            positions,
                            ensure_ascii=False
                        )
                    }
                ],
                "temperature": 0
            }
        )

        result = response.json()

        content = result["choices"][0]["message"]["content"]

        try:
            parsed = json.loads(content)

        except Exception as e:
            print("JSON parse error:", e)

            start = content.find("[")
            end = content.rfind("]") + 1

            parsed = json.loads(content[start:end])

        return parsed

    except Exception as e:
        print("API Error:", e)
        return []


def get_unmapped_positions(df):
    """
    Get only positions that are not yet normalized
    """

    if "new_position" not in df.columns:
        df["new_position"] = None

    mask = (
        df["new_position"].isna()
        | (df["new_position"] == "")
    )

    unmapped_df = df[mask]

    positions = (
        unmapped_df["cleaned_position"]
        .dropna()
        .astype(str)
        .str.strip()
        .unique()
        .tolist()
    )

    return positions, mask


def apply_results_to_df(df, results):
    """
    Apply normalized positions to dataframe
    """

    print("=" * 50)
    print("results are:", results)
    print("=" * 50)

    if not isinstance(results, list):
        return df

    # normalize results map
    normalized_map = {}

    for item in results:

        if not isinstance(item, dict):
            continue

        position = item.get("position")

        if not position:
            continue

        normalized_key = str(position).strip().lower()

        normalized_map[normalized_key] = item

    # apply to dataframe
    for idx, row in df.iterrows():

        position = row.get("cleaned_position")

        if pd.isna(position):
            continue

        normalized_position = str(position).strip().lower()

        if normalized_position in normalized_map:

            matched_item = normalized_map[normalized_position]

            df.at[idx, "new_position"] = matched_item.get(
                "new_position"
            )

    return df


def batch_process(
        positions,
        df,
        batch_size=50,
        output_file="filtered_positions.csv"
):
    """
    Process positions in batches
    """

    total_batches = math.ceil(
        len(positions) / batch_size
    )

    for i in range(0, len(positions), batch_size):

        batch_num = (i // batch_size) + 1

        batch = positions[i:i + batch_size]

        print("=" * 50)
        print(
            f"Processing batch "
            f"{batch_num} / {total_batches}"
        )
        print("=" * 50)

        result = classifier(batch)

        df = apply_results_to_df(df, result)

        # incremental save
        df.to_csv(output_file, index=False)

        print(
            f"✅ Saved progress after batch "
            f"{batch_num}"
        )

        # avoid rate limiting
        time.sleep(1)

    return df


def sector_subsector_mapper(test_mode=True):

    output_file = "filtered_positions.csv"

    # load source file
    df_positions = pd.read_excel(
        "all_positions_12k.xlsx"
    )

    # resume previous progress if exists
    if os.path.exists(output_file):

        print("Loading existing progress file...")

        df_positions = pd.read_csv(output_file)

    # get only unmapped positions
    positions, mask = get_unmapped_positions(
        df_positions
    )

    print(f"Total unique positions: {len(positions)}")

    if test_mode:

        print("=" * 50)
        print("RUNNING TEST MODE")
        print("=" * 50)

        sample_positions = positions[:10]

        results = classifier(sample_positions)

        df_positions = apply_results_to_df(
            df_positions,
            results
        )

        test_output = "filtered_positions_test.csv"

        df_positions.to_csv(
            test_output,
            index=False
        )

        print(
            f"✅ Test output saved to "
            f"{test_output}"
        )

    else:

        print("=" * 50)
        print("RUNNING PRODUCTION MODE")
        print("=" * 50)

        df_positions = batch_process(
            positions=positions,
            df=df_positions,
            batch_size=50,
            output_file=output_file
        )

        print(
            f"✅ Production output saved to "
            f"{output_file}"
        )


if __name__ == "__main__":

    # change to True for testing
    sector_subsector_mapper(
        test_mode=False
    )