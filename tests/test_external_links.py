"""Homepage cache safety, external-link identity, and social rule behavior."""

import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from utils.external_links import (
    LINK_COLUMNS, classify_social_links, extract_external_links, extract_homepage_links,
    load_social_rules, read_homepage,
)
from utils.scraping import get_page


class ExternalLinksTests(unittest.TestCase):
    def test_all_links_preserve_internal_school_pages_and_subdomains(self):
        html = '''<a href="/school#about">Parish school</a>
        <a href="https://school.parish.org/admissions">Admissions</a>
        <a href="https://other-school.edu/">Partner school</a>
        <a href="/school#contact">School contact</a>'''
        rows = extract_homepage_links(html, "https://parish.org/", "https://www.parish.org/")
        self.assertEqual([row["link_type"] for row in rows], ["internal", "internal", "external"])
        self.assertEqual(rows[0]["link_url"], "https://www.parish.org/school")
        self.assertEqual(rows[0]["link_count"], 2)
        external = extract_external_links(html, "https://parish.org/", "https://www.parish.org/")
        self.assertEqual(len(external), 1)
        self.assertEqual(external[0]["external_url"], "https://other-school.edu/")

    def test_redirect_base_internal_hosts_and_duplicate_labels(self):
        html = '''<base href="https://school.edu/about/">
        <a href="/admissions#apply">School</a>
        <a href="https://school.edu/admissions#more"><img alt="Admissions"></a>
        <a href="//www.facebook.com/OurParish#top" aria-label="Facebook"></a>
        <a href="https://newparish.org/news">Internal</a>
        <a href="https://school.oldparish.org/">Internal subdomain</a>
        <a href="mailto:office@parish.org">Email</a>
        <a href="tel:5551234">Phone</a><a href="#top">Top</a>
        <a href="javascript:alert(1)">JS</a>
        <a href="http://[bad">Malformed</a>
        <area href="https://other.edu/map" alt="Campus">
        '''
        rows = extract_external_links(html, "https://oldparish.org", "https://www.newparish.org/")
        self.assertEqual(len(rows), 3)
        self.assertEqual(rows[0]["external_url"], "https://school.edu/admissions")
        self.assertEqual(rows[0]["link_count"], 2)
        self.assertEqual(rows[0]["link_text"], "School | Admissions")
        self.assertEqual(rows[1]["external_host"], "facebook.com")

    def test_cache_only_never_requests_and_accepts_discovery_key(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch("utils.scraping.requests.Session") as session:
                with self.assertRaises(FileNotFoundError):
                    get_page("https://parish.org/", folder, cache_only=True)
                session.assert_not_called()
                key = sha256(b"https://parish.org/").hexdigest()
                (Path(folder) / f"{key}.html").write_bytes(b"<html>Cached</html>")
                (Path(folder) / f"{key}.url").write_text("https://newparish.org/", encoding="utf-8")
                html, final = read_homepage("https://parish.org", folder)
                self.assertEqual(html, b"<html>Cached</html>")
                self.assertEqual(final, "https://newparish.org/")
                session.assert_not_called()

    def test_rule_boundaries_exclusions_and_repeat_application(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "rules.csv"
            path.write_text(
                "match_type,pattern,is_social,service,category\n"
                "domain_path_prefix,facebook.com/sharer,FALSE,Share,social_media\n"
                "domain,facebook.com,TRUE,Facebook,social_media\n"
                "domain,flocknote.com,TRUE,Flocknote,parish_communication\n",
                encoding="utf-8",
            )
            rules = load_social_rules(path)
        links = pd.DataFrame({"external_url": [
            "https://www.facebook.com/parish", "https://facebook.com/sharer.php?u=x",
            "https://parish.flocknote.com/", "https://facebook.com.evil.org/",
            "https://notfacebook.com/", "https://school.edu/",
        ]})
        result = classify_social_links(links, rules)
        self.assertEqual(result["is_social"].tolist(), [True, False, True, False, False, False])
        self.assertEqual(len(result), len(links))
        pd.testing.assert_frame_equal(classify_social_links(result, rules), result)

    def test_default_rules_cover_requested_services_and_empty_inputs(self):
        rules = load_social_rules("data/04_extract_features/social_link_rules.csv")
        urls = ["https://youtube.com/parish", "https://parish.flocknote.com/",
                "https://parish.us1.list-manage.com/subscribe?u=1",
                "https://visitor.r20.constantcontact.com/manage/optin"]
        result = classify_social_links(pd.DataFrame({"external_url": urls}), rules)
        self.assertTrue(result["is_social"].all())
        empty = classify_social_links(pd.DataFrame(columns=LINK_COLUMNS), rules)
        self.assertEqual(len(empty), 0)
        self.assertEqual(empty["is_social"].dtype, bool)


if __name__ == "__main__":
    unittest.main()
