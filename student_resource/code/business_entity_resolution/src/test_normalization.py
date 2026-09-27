import unittest
import pandas as pd
from normalization import normalize_single_record, normalize_dataframe

class TestNormalization(unittest.TestCase):

    def test_mojibake_repair(self):
        bname, addr, nl = normalize_single_record("Gali No. Â€“ 01", "Vspâ€™S Bhavana", "India")
        self.assertEqual(bname, "gali no 01")
        self.assertEqual(addr, "vsp s bhavana")
        self.assertFalse(nl)

    def test_non_latin_detection(self):
        bname, addr, nl = normalize_single_record("राम मार्केटिंग प्राइवेट लिमिटेड", "KH NO. -570/13", "India")
        self.assertEqual(bname, "")
        self.assertEqual(addr, "kh no 570 13")
        self.assertTrue(nl)

    def test_sa_sas_suffix_country_scoped(self):
        bname_us, _, _ = normalize_single_record("SA Adamas LLC", "Addr", "US")
        self.assertEqual(bname_us, "sa adamas")
        
        bname_in, _, _ = normalize_single_record("SAS Educational Society", "Addr", "India")
        self.assertEqual(bname_in, "sas educational society")
        
        bname_fr, _, _ = normalize_single_record("Total SA", "Addr", "France")
        self.assertEqual(bname_fr, "total")

    def test_anchored_end_and_stacked_suffixes(self):
        # Confirm Private / Limited at start are NOT stripped when part of the business name
        b1, _, _ = normalize_single_record("Private Wealth Advisors", "Addr", "US")
        self.assertEqual(b1, "private wealth advisors")
        
        b2, _, _ = normalize_single_record("Limited Edition Salon", "Addr", "US")
        self.assertEqual(b2, "limited edition salon")
        
        # Stacked suffixes at end "Pvt Ltd" -> stripped to "acme"
        b3, _, _ = normalize_single_record("Acme Pvt Ltd", "Addr", "India")
        self.assertEqual(b3, "acme")

    def test_domain_and_markdown(self):
        bname, _, _ = normalize_single_record("Ara Aero Private Limited", "2499 Office.Com", "India")
        self.assertEqual(bname, "ara aero")
        
        bname_md, _, _ = normalize_single_record("Company Name | [site](https://site.com)", "123 Main St", "US")
        self.assertEqual(bname_md, "company name site")

    def test_dba_trade_name(self):
        # Leading DBA marker stripped -> "construction"
        bname_dba_start, _, _ = normalize_single_record("Dba Construction Pvt Ltd", "Addr", "India")
        self.assertEqual(bname_dba_start, "construction")
        
        # Mid-string DBA trade name -> left side "acme"
        bname_dba_mid, _, _ = normalize_single_record("Acme Corp dba Apex Solutions", "Addr", "US")
        self.assertEqual(bname_dba_mid, "acme")

    def test_leetspeak_and_address_null(self):
        bname, addr, _ = normalize_single_record("5uperior Quality Products LLC", "Doorno.52&53, 9Thfloorcollegeroad", "India")
        self.assertEqual(bname, "superior quality products")
        self.assertEqual(addr, "doorno 52 53 9 th floorcollegeroad")
        
        _, addr_null, _ = normalize_single_record("Name", "NULL", "US")
        self.assertEqual(addr_null, "")

    def test_dataframe_batch(self):
        df = pd.DataFrame({
            'business_name': ['SA Adamas LLC', '5uperior Products'],
            'business_address': ['100 Main St', 'NULL'],
            'country': ['US', 'US']
        })
        df_out = normalize_dataframe(df)
        self.assertEqual(list(df_out['norm_name']), ['sa adamas', 'superior products'])
        self.assertEqual(list(df_out['norm_address']), ['100 main st', ''])

if __name__ == '__main__':
    unittest.main()
