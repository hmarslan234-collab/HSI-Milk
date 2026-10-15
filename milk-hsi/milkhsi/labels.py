"""Labels for each capture.

The manifest (data/manifest.csv) has one row per capture folder:

    capture, species, adulterant, level, sample_id

* species     cow | buffalo | goat
* adulterant  pure | water | salicylic | boric | glucose | formalin
* level       0 for pure; 1..10 for the ten levels of each adulterant
              (water 4,8,...,40 %  ->  1..10 ; chemicals 1..10 mL/100 mL -> 1..10)
* sample_id   the physical sample. Usually equal to `capture`. If the same
              dish was scanned more than once, give the repeats the same
              sample_id so they stay on the same side of every split.

`guess_from_name` fills these columns from the folder name; check the CSV by
hand before running the models.
"""
from __future__ import annotations

import re

import pandas as pd

SPECIES = ["cow", "buffalo", "goat"]
ADULTERANTS = ["pure", "water", "salicylic", "boric", "glucose", "formalin"]
WATER_LEVELS = [4, 8, 12, 16, 20, 24, 28, 32, 36, 40]


def guess_from_name(name: str) -> dict:
    n = name.lower()
    species = next((s for s in SPECIES if s in n), "")
    adulterant = next((a for a in ADULTERANTS[1:] if a in n), "")
    if not adulterant:
        adulterant = "pure" if ("pure" in n or not re.search(r"\d", n)) else ""
    level = 0
    if adulterant not in ("pure", ""):
        nums = [int(x) for x in re.findall(r"\d+", n)]
        if nums:
            v = nums[0]
            if adulterant == "water" and v in WATER_LEVELS:
                level = WATER_LEVELS.index(v) + 1
            else:
                level = v
    return {"capture": name, "species": species, "adulterant": adulterant,
            "level": level, "sample_id": name}


def add_task_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["binary"] = (df["adulterant"] != "pure").map({False: "pure", True: "adulterated"})
    df["level"] = df["level"].astype(int)
    df["level_class"] = df.apply(
        lambda r: "pure" if r["adulterant"] == "pure" else f"{r['adulterant']}_{int(r['level']):02d}", axis=1)
    return df


def task_target(df: pd.DataFrame, task: str) -> pd.Series:
    """Column used as the target for a task name from config.tasks."""
    return {"species": df["species"], "binary": df["binary"],
            "adulterant": df["adulterant"], "level": df["level_class"]}[task]
