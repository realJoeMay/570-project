"""Look up parish counties using the US Census Geocoder."""

import json
import os
import re
from functools import lru_cache
from hashlib import sha256
from io import BytesIO
from pathlib import Path
from zipfile import BadZipFile, ZipFile

import pandas as pd
import requests
from dotenv import dotenv_values


GEOCODER_URL = "https://geocoding.geo.census.gov/geocoder/geographies/onelineaddress"
POSTAL_URL = "https://download.geonames.org/export/zip/US.zip"
STATE_FIPS = dict(pair.split(":") for pair in (
    "AL:01 AK:02 AZ:04 AR:05 CA:06 CO:08 CT:09 DE:10 DC:11 FL:12 GA:13 "
    "HI:15 ID:16 IL:17 IN:18 IA:19 KS:20 KY:21 LA:22 ME:23 MD:24 MA:25 "
    "MI:26 MN:27 MS:28 MO:29 MT:30 NE:31 NV:32 NH:33 NJ:34 NM:35 NY:36 "
    "NC:37 ND:38 OH:39 OK:40 OR:41 PA:42 RI:44 SC:45 SD:46 TN:47 TX:48 "
    "UT:49 VT:50 VA:51 WA:53 WV:54 WI:55 WY:56"
).split())


def get_county_cache_dir(cache_dir: str | Path | None = None) -> Path:
    """Resolve an override, COUNTY_CACHE_DIR from environment/.env, or default."""
    if cache_dir is None:
        cache_dir = (
            os.environ.get("COUNTY_CACHE_DIR")
            or dotenv_values(".env").get("COUNTY_CACHE_DIR")
            or "data/cache/census_geocoder"
        )
    return Path(cache_dir).expanduser()


@lru_cache(maxsize=4)
def _load_postal_data(cache_dir: str) -> pd.DataFrame:
    """Download the GeoNames reference once and preserve the original ZIP."""
    path = Path(cache_dir) / "geonames_US.zip"
    if path.exists():
        content = path.read_bytes()
    else:
        response = requests.get(POSTAL_URL, timeout=60)
        response.raise_for_status()
        content = response.content
    with ZipFile(BytesIO(content)) as archive:
        with archive.open("US.txt") as file:
            data = pd.read_csv(
                file, sep="\t", header=None, dtype=str, keep_default_na=False,
                usecols=[1, 2, 3, 4, 5, 6],
            )
    data.columns = ["zip", "city", "state_name", "state", "county", "county_code"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return data


def _normalize_city(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _lookup_locality(address: str, cache_dir: str | Path) -> dict:
    """Try city/state, then ZIP; accept only one county in the reference rows."""
    postal = _load_postal_data(str(Path(cache_dir)))
    zip_match = re.search(r"\b(\d{5})(?:-\d{4})?\s*$", address)
    zip_code = zip_match.group(1) if zip_match else None
    location = address[:zip_match.start()].rstrip(" ,") if zip_match else address
    state_names = dict(zip(postal["state_name"].str.casefold(), postal["state"]))
    state_names.update({state.casefold(): state for state in STATE_FIPS})
    state_pattern = "|".join(re.escape(name) for name in sorted(state_names, key=len, reverse=True))
    state_match = re.search(rf"(?:^|[,\s])({state_pattern})\s*$", location, re.IGNORECASE)
    state = state_names[state_match.group(1).casefold()] if state_match else None
    city = location[:state_match.start()].rstrip(" ,").rsplit(",", 1)[-1].strip() if state_match else ""
    queries = []
    if city and state and not any(character.isdigit() for character in city):
        rows = postal.loc[
            (postal["state"] == state)
            & (postal["city"].map(_normalize_city) == _normalize_city(city))
        ]
        queries.append(("city_state", f"{city}, {state}", rows))
    if zip_code:
        rows = postal.loc[postal["zip"] == zip_code]
        if state:
            rows = rows.loc[rows["state"] == state]
        queries.append(("zip", zip_code, rows))

    status = "no_match"
    for method, query, rows in queries:
        if rows.empty:
            continue
        valid = (
            rows["state"].isin(STATE_FIPS)
            & rows["county_code"].str.fullmatch(r"\d{1,3}")
            & rows["county"].ne("")
        )
        if not valid.all():
            status = "no_county" if status == "no_match" else status
            continue
        fips = rows["state"].map(STATE_FIPS) + rows["county_code"].str.zfill(3)
        if fips.nunique() != 1:
            status = "ambiguous"
            continue
        return {
            "county": rows.iloc[0]["county"],
            "county_fips": fips.iloc[0],
            "county_lookup_status": "matched",
            "county_lookup_method": method,
            "county_matched_address": query,
        }
    return {"county_lookup_status": status}


def lookup_county(
    address: str,
    cache_dir: str | Path | None = None,
) -> dict:
    """Return county details and a review status; cache unmodified API responses.

    Census no-match responses (including cached ones) fall back to GeoNames
    city/state, then ZIP. Locality matches are approximate and marked by method.
    Multiple Census address matches do not trigger fallback. Request failures
    are not cached, so a later run can retry them.
    Cache location follows COUNTY_CACHE_DIR unless explicitly overridden.
    """
    result = {
        "county": None,
        "county_fips": None,
        "county_lookup_status": "missing_address",
        "county_lookup_method": None,
        "county_matched_address": None,
        "county_lookup_error": None,
    }
    address = address.strip()
    if not address:
        return result
    cache_dir = get_county_cache_dir(cache_dir)

    params = {
        "address": address,
        "benchmark": "Public_AR_Current",
        "vintage": "Current_Current",
        "format": "json",
    }
    cache_key = sha256(json.dumps(params, sort_keys=True).encode("utf-8")).hexdigest()
    cache_path = Path(cache_dir) / f"{cache_key}.json"
    try:
        if cache_path.exists():
            payload = json.loads(cache_path.read_bytes())
        else:
            response = requests.get(GEOCODER_URL, params=params, timeout=30)
            response.raise_for_status()
            payload = response.json()
            # Validate the response before preserving the original response bytes.
            payload["result"]["addressMatches"]
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(response.content)
        matches = payload["result"]["addressMatches"]
        if not matches:
            result.update(_lookup_locality(address, cache_dir))
        elif len(matches) != 1:
            result["county_lookup_status"] = "ambiguous"
        else:
            match = matches[0]
            counties = match.get("geographies", {}).get("Counties", [])
            result["county_matched_address"] = match.get("matchedAddress")
            if len(counties) == 1:
                county = counties[0]
                result.update(
                    county=county["NAME"],
                    county_fips=county["GEOID"],
                    county_lookup_status="matched",
                    county_lookup_method="address",
                )
            else:
                result["county_lookup_status"] = "ambiguous" if counties else "no_county"
    except (requests.RequestException, ValueError, KeyError, TypeError, BadZipFile) as exc:
        result.update(county_lookup_status="error", county_lookup_error=str(exc))
    return result
