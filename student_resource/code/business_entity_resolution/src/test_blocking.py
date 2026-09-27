import unittest
from blocking import extract_blocking_keys, build_inverted_index, generate_candidates_for_entity

class TestBlocking(unittest.TestCase):

    def test_extract_blocking_keys_latin(self):
        keys = extract_blocking_keys("superio quality products", "100 main st 110037", "US")
        self.assertTrue(any(k.startswith("N1:US:") for k in keys))
        self.assertTrue(any(k.startswith("PIN:US:110037") for k in keys))

    def test_extract_blocking_keys_non_latin(self):
        # Non-Latin name yields no name keys, but address keys are extracted
        keys = extract_blocking_keys("", "kh no 570 13 new delhi 110037", "India")
        self.assertFalse(any(k.startswith("N1:") for k in keys))
        self.assertTrue(any(k.startswith("A_") or k.startswith("PIN:") for k in keys))

    def test_inverted_index_and_candidate_generation(self):
        records = [
            ("S2_101", "superio quality products", "100 main st 110037", "US"),
            ("S3_202", "ram marketing", "570 new delhi 110037", "India")
        ]
        index = build_inverted_index(records)
        self.assertGreater(len(index), 0)
        
        cands = generate_candidates_for_entity("ram marketing", "570 new delhi 110037", "India", index)
        self.assertIn("S3_202", cands)

if __name__ == '__main__':
    unittest.main()
