"""County lookup acceptance, caching, and failure handling."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import requests
import pandas as pd

from utils.county_lookup import get_county_cache_dir, lookup_county


class CountyLookupTests(unittest.TestCase):
    def test_cache_location_precedence(self):
        with patch.dict("os.environ", {"COUNTY_CACHE_DIR": "environment/census"}), patch(
            "utils.county_lookup.dotenv_values", return_value={"COUNTY_CACHE_DIR": "dotenv/census"},
        ):
            self.assertEqual(get_county_cache_dir(), Path("environment/census"))
            self.assertEqual(get_county_cache_dir("explicit/census"), Path("explicit/census"))
        with patch.dict("os.environ", {}, clear=True), patch(
            "utils.county_lookup.dotenv_values", return_value={"COUNTY_CACHE_DIR": "dotenv/census"},
        ):
            self.assertEqual(get_county_cache_dir(), Path("dotenv/census"))
        with patch.dict("os.environ", {}, clear=True), patch("utils.county_lookup.dotenv_values", return_value={}):
            self.assertEqual(get_county_cache_dir(), Path("data/cache/census_geocoder"))

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        postal = pd.DataFrame([
            ["36611", "Chickasaw", "Alabama", "AL", "Mobile", "097"],
            ["35071", "Warrior", "Alabama", "AL", "Jefferson", "073"],
            ["35180", "Warrior", "Alabama", "AL", "Blount", "009"],
        ], columns=["zip", "city", "state_name", "state", "county", "county_code"])
        loader = patch("utils.county_lookup._load_postal_data", return_value=postal)
        self.postal_loader = loader.start()
        self.addCleanup(loader.stop)

    def response(self, matches):
        payload = {"result": {"addressMatches": matches}}
        return Mock(json=Mock(return_value=payload), content=json.dumps(payload).encode())

    def test_match_preserves_fips_and_reuses_raw_response(self):
        match = {
            "matchedAddress": "123 MAIN ST",
            "geographies": {"Counties": [{"NAME": "Autauga County", "GEOID": "01001"}]},
        }
        with patch("utils.county_lookup.requests.get", return_value=self.response([match])) as get:
            first = lookup_county("123 Main St", self.folder.name)
            second = lookup_county("123 Main St", self.folder.name)
        self.assertEqual(first["county_fips"], "01001")
        self.assertEqual(first["county_lookup_status"], "matched")
        self.assertEqual(first, second)
        get.assert_called_once()
        self.postal_loader.assert_not_called()
        self.assertEqual(first["county_lookup_method"], "address")

    def test_city_state_fallback_including_cached_no_match(self):
        with patch("utils.county_lookup.requests.get", return_value=self.response([])) as get:
            first = lookup_county("Chickasaw, AL", self.folder.name)
            second = lookup_county("Chickasaw, AL", self.folder.name)
        self.assertEqual(first["county_fips"], "01097")
        self.assertEqual(first["county_lookup_method"], "city_state")
        self.assertEqual(first, second)
        get.assert_called_once()

    def test_ambiguous_city_retries_zip_with_full_state_and_zip_plus_four(self):
        with patch("utils.county_lookup.requests.get", return_value=self.response([])):
            result = lookup_county("221 5th Street, Warrior, Alabama, 35071-1234", self.folder.name)
        self.assertEqual(result["county_fips"], "01073")
        self.assertEqual(result["county_lookup_method"], "zip")

    def test_zip_only_fallback_and_unknown_city(self):
        for address in ["36611", "123 Missing St, Unknown City, AL, 36611"]:
            with self.subTest(address=address):
                with patch("utils.county_lookup.requests.get", return_value=self.response([])):
                    result = lookup_county(address, self.folder.name)
                self.assertEqual(result["county_fips"], "01097")
                self.assertEqual(result["county_lookup_method"], "zip")

    def test_ambiguous_locality_remains_unresolved(self):
        for address in ["Warrior, AL", "Missing City, AL, 99999"]:
            with self.subTest(address=address):
                with patch("utils.county_lookup.requests.get", return_value=self.response([])):
                    result = lookup_county(address, self.folder.name)
                self.assertIsNone(result["county_fips"])
                expected = "ambiguous" if address == "Warrior, AL" else "no_match"
                self.assertEqual(result["county_lookup_status"], expected)

    def test_ambiguous_census_match_does_not_use_locality(self):
        with patch("utils.county_lookup.requests.get", return_value=self.response([{}, {}])):
            result = lookup_county("Chickasaw, AL", self.folder.name)
        self.assertEqual(result["county_lookup_status"], "ambiguous")
        self.postal_loader.assert_not_called()

    def test_unresolved_addresses_do_not_get_counties(self):
        for matches, status in [([], "no_match"), ([{}, {}], "ambiguous"), ([{}], "no_county")]:
            with self.subTest(status=status), tempfile.TemporaryDirectory() as folder:
                with patch("utils.county_lookup.requests.get", return_value=self.response(matches)):
                    result = lookup_county("Unresolved address", folder)
                self.assertEqual(result["county_lookup_status"], status)
                self.assertIsNone(result["county_fips"])

    def test_failure_is_retryable(self):
        with patch("utils.county_lookup.requests.get", side_effect=requests.Timeout("Timed out")) as get:
            for _ in range(2):
                result = lookup_county("123 Main St", self.folder.name)
                self.assertEqual(result["county_lookup_status"], "error")
            self.assertEqual(get.call_count, 2)

    def test_empty_address_does_not_request(self):
        with patch("utils.county_lookup.requests.get") as get:
            self.assertEqual(lookup_county("  ", self.folder.name)["county_lookup_status"], "missing_address")
        get.assert_not_called()


if __name__ == "__main__":
    unittest.main()
