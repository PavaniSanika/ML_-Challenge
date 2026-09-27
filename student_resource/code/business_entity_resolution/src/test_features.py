import unittest
from features import extract_candidate_pair_features, FEATURE_NAMES, get_script_type

class TestFeatures(unittest.TestCase):

    def test_latin_pair_features(self):
        rec1 = {"norm_name": "acme corp", "norm_address": "100 broadway ny 10001", "country": "US", "is_non_latin": False, "entity_id": "S1_001"}
        rec2 = {"norm_name": "acme corporation", "norm_address": "100 broadway ny 10001", "country": "US", "is_non_latin": False, "entity_id": "S2_101"}
        
        feats = extract_candidate_pair_features(rec1, rec2)
        self.assertEqual(feats["is_script_mismatch"], 0.0)
        self.assertEqual(feats["both_latin"], 1.0)
        self.assertGreater(feats["name_jaccard"], 0.0)
        self.assertEqual(feats["pincode_exact_match"], 1.0)
        self.assertEqual(feats["has_address_both"], 1.0)

    def test_tamil_vs_devanagari_script_mismatch(self):
        # Tamil vs Devanagari script mismatch: name features must be 0.0
        rec1 = {"raw_name": "\u0b95\u0bc1\u0bb3\u0bcb\u0baa\u0bb2\u0bcd \u0baa\u0bbf\u0b9a\u0bbf\u0ba9\u0bb8\u0bcd \u0baa\u0bbf\u0bb0\u0bc8\u0bb5\u0bc7\u0b9f\u0bcd \u0bb2\u0bbf\u0bae\u0bbf\u0b9f\u0bc6\u0b9f\u0bcd", "norm_name": "tamil name", "norm_address": "ambattur chennai 600053", "country": "India", "is_non_latin": True, "entity_id": "S1_002"}
        rec2 = {"raw_name": "\u0917\u094d\u0932\u094b\u092c\u0932 \u092c\u093f\u091c\u0928\u0947\u0938 \u092a\u094d\u0930\u093e\u0907\u0935\u0947\u091f \u0932\u093f\u092e\u093f\u091f\u0947\u0921", "norm_name": "devanagari name", "norm_address": "ambattur chennai 600053", "country": "India", "is_non_latin": True, "entity_id": "S2_202"}
        
        feats = extract_candidate_pair_features(rec1, rec2)
        self.assertEqual(feats["is_script_mismatch"], 1.0)
        self.assertEqual(feats["same_non_latin_script"], 0.0)
        self.assertEqual(feats["name_jaccard"], 0.0)
        self.assertEqual(feats["name_char_3gram_sim"], 0.0)
        self.assertEqual(feats["has_address_both"], 1.0)
        self.assertGreater(feats["address_jaccard"], 0.5)

    def test_kannada_vs_telugu_script_mismatch(self):
        # Kannada vs Telugu script mismatch
        rec1 = {"raw_name": "\u0c95\u0cb0\u0ccd\u0ca8\u0cbe\u0c9f\u0c95 \u0c89\u0ca6\u0ccd\u0caf\u0cae", "norm_name": "kannada text", "norm_address": "mg road bengaluru 560001", "country": "India", "is_non_latin": True, "entity_id": "S1_003"}
        rec2 = {"raw_name": "\u0c24\u0c46\u0c32\u0c02\u0c17\u0c3e\u0c23 \u0c38\u0c02\u0c38\u0c4d\u0c25", "norm_name": "telugu text", "norm_address": "mg road bengaluru 560001", "country": "India", "is_non_latin": True, "entity_id": "S2_303"}
        
        feats = extract_candidate_pair_features(rec1, rec2)
        self.assertEqual(get_script_type(rec1["raw_name"]), "kannada")
        self.assertEqual(get_script_type(rec2["raw_name"]), "telugu")
        self.assertEqual(feats["is_script_mismatch"], 1.0)
        self.assertEqual(feats["both_non_latin"], 1.0)
        self.assertEqual(feats["same_non_latin_script"], 0.0)

    def test_kannada_same_script_match(self):
        # Kannada vs Kannada (same script match)
        rec1 = {"raw_name": "\u0c95\u0cb0\u0ccd\u0ca8\u0cbe\u0c9f\u0c95 \u0c89\u0ca6\u0ccd\u0caf\u0cae \u0ca8\u0cbf\u0caf\u0cae\u0cbf\u0ca4", "norm_name": "kannada name 1", "norm_address": "bengaluru 560001", "country": "India", "is_non_latin": True, "entity_id": "S1_004"}
        rec2 = {"raw_name": "\u0c95\u0cb0\u0ccd\u0ca8\u0cbe\u0c9f\u0c95 \u0c89\u0ca6\u0ccd\u0caf\u0cae \u0cb8\u0c82\u0cb8\u0ccd\u0ca5\u0cc6", "norm_name": "kannada name 2", "norm_address": "bengaluru 560001", "country": "India", "is_non_latin": True, "entity_id": "S2_404"}
        
        feats = extract_candidate_pair_features(rec1, rec2)
        self.assertEqual(get_script_type(rec1["raw_name"]), "kannada")
        self.assertEqual(get_script_type(rec2["raw_name"]), "kannada")
        self.assertEqual(feats["is_script_mismatch"], 0.0)
        self.assertEqual(feats["both_non_latin"], 1.0)
        self.assertEqual(feats["same_non_latin_script"], 1.0)

    def test_bengali_vs_devanagari_script_mismatch(self):
        # Bengali vs Devanagari script mismatch
        rec1 = {"raw_name": "\u09aa\u09b6\u09cd\u099a\u09bf\u09ae\u09ac\u0999\u09cd\u0997 \u09ac\u09cd\u09af\u09ac\u09b8\u09be", "norm_name": "bengali name", "norm_address": "kolkata 700001", "country": "India", "is_non_latin": True, "entity_id": "S1_005"}
        rec2 = {"raw_name": "\u092a\u0936\u094d\u091a\u093f\u092e \u092c\u0902\u0917\u093e\u0932 \u0935\u094d\u092f\u093e\u092a\u093e\u0930", "norm_name": "devanagari name", "norm_address": "kolkata 700001", "country": "India", "is_non_latin": True, "entity_id": "S2_505"}
        
        feats = extract_candidate_pair_features(rec1, rec2)
        self.assertEqual(get_script_type(rec1["raw_name"]), "bengali")
        self.assertEqual(get_script_type(rec2["raw_name"]), "devanagari")
        self.assertEqual(feats["is_script_mismatch"], 1.0)
        self.assertEqual(feats["same_non_latin_script"], 0.0)

    def test_empty_address_handling(self):
        # Empty address on one side -> has_address_both = 0.0, all address sim features 0.0
        rec1 = {"norm_name": "acme corp", "norm_address": "100 broadway ny", "country": "US", "is_non_latin": False, "entity_id": "S1_003"}
        rec2 = {"norm_name": "acme corp", "norm_address": "", "country": "US", "is_non_latin": False, "entity_id": "S2_303"}
        
        feats = extract_candidate_pair_features(rec1, rec2)
        self.assertEqual(feats["has_address_both"], 0.0)
        self.assertEqual(feats["address_jaccard"], 0.0)
        self.assertEqual(feats["address_containment"], 0.0)
        self.assertEqual(feats["address_edit_ratio"], 0.0)
        self.assertEqual(feats["digit_jaccard"], 0.0)

    def test_all_feature_names_present(self):
        rec1 = {"norm_name": "test", "norm_address": "123 main", "country": "US", "is_non_latin": False, "entity_id": "S1_1"}
        rec2 = {"norm_name": "test", "norm_address": "123 main", "country": "US", "is_non_latin": False, "entity_id": "S2_2"}
        feats = extract_candidate_pair_features(rec1, rec2)
        for fname in FEATURE_NAMES:
            self.assertIn(fname, feats)

if __name__ == '__main__':
    unittest.main()