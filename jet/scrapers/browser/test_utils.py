import unittest
from jet.scrapers.browser.formatters import construct_browser_query


class TestConstructBrowserQuery(unittest.TestCase):

    def test_valid_query(self):
        query = construct_browser_query(
            search_terms="top 10 romantic comedy anime",
            include_sites=["myanimelist.net", "animenewsnetwork.com"],
            exclude_sites=["wikipedia.org", "imdb.com"],
            after_date="2025-01-01",
            before_date="2025-04-05"
        )
        expected_query = (
            "top 10 romantic comedy anime "
            "-site:wikipedia.org -site:imdb.com "
            "site:myanimelist.net site:animenewsnetwork.com "
            "after:2025-01-01 before:2025-04-05"
        )
        self.assertEqual(query, expected_query)

    def test_empty_search_terms(self):
        with self.assertRaises(ValueError):
            construct_browser_query(
                search_terms="",
                include_sites=["myanimelist.net"],
                exclude_sites=["wikipedia.org"]
            )

    def test_invalid_date_format(self):
        with self.assertRaises(ValueError):
            construct_browser_query(
                search_terms="top 10 romantic comedy anime",
                include_sites=["myanimelist.net"],
                exclude_sites=["wikipedia.org"],
                after_date="2025-01-32",  # Invalid date
                before_date="2025-04-05"
            )

    def test_invalid_site_format_in_include(self):
        with self.assertRaises(ValueError):
            construct_browser_query(
                search_terms="top 10 romantic comedy anime",
                # Invalid site format
                include_sites=["invalidsite.com", "another*site.com"],
                exclude_sites=["wikipedia.org"]
            )

    def test_invalid_site_format_in_exclude(self):
        with self.assertRaises(ValueError):
            construct_browser_query(
                search_terms="top 10 romantic comedy anime",
                include_sites=["myanimelist.net"],
                exclude_sites=["invalid*site.com",
                               "imdb.com"]  # Invalid site format
            )

    def test_no_date_filter(self):
        query = construct_browser_query(
            search_terms="top 10 romantic comedy anime",
            include_sites=["myanimelist.net"],
            exclude_sites=["wikipedia.org"]
        )
        expected_query = (
            "top 10 romantic comedy anime "
            "-site:wikipedia.org "
            "site:myanimelist.net"
        )
        self.assertEqual(query, expected_query)


if __name__ == '__main__':
    unittest.main()
