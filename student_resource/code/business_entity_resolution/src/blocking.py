"""
Candidate Generation / Blocking Module for Business Entity Resolution.
Implements dual-path blocking (Name-based + Address-based).
Address blocking ensures candidates are generated for Non-Latin script entities
where name similarity is ~0.
"""

import re
from collections import defaultdict

NAME_STOPWORDS = {
    "group", "services", "solutions", "systems", "enterprises", "holdings",
    "consulting", "tech", "global", "center", "centre", "international",
    "industries", "associates", "agency", "management", "ventures", "traders",
    "trading", "logistics", "infotech", "technologies", "products", "works",
    "company", "corp", "pvt", "ltd", "inc", "llc", "co", "private", "limited",
    "public", "society", "trust", "foundation", "development", "developers"
}

ADDR_STOPWORDS = {
    "street", "st", "road", "rd", "avenue", "ave", "lane", "ln", "drive", "dr",
    "boulevard", "blvd", "court", "ct", "place", "pl", "way", "highway", "hwy",
    "suite", "ste", "unit", "apt", "apartment", "building", "bldg", "floor", "fl",
    "flat", "no", "number", "plot", "near", "opp", "opposite", "behind", "dist",
    "district", "tehsil", "taluka", "nagar", "colony", "marg", "society", "delhi",
    "mumbai", "new", "box", "po", "pob", "house", "room", "tower", "block", "kh",
    "shp", "shop", "villa", "bungalow", "survey", "sy", "s", "no", "sec", "sector",
    "phase", "stage", "main", "cross", "sub", "via", "pvt", "ltd", "g", "ff", "gf",
    "floor", "first", "second", "third", "ground", "c", "o", "care", "of"
}

RE_CLEAN = re.compile(r'[^a-zA-Z0-9\s]')
RE_PINCODE = re.compile(r'\b\d{5,6}\b')

def is_non_latin_text(text: str) -> bool:
    if not text:
        return False
    non_lat = sum(1 for ch in text if ord(ch) > 0x024F)
    return (non_lat / len(text)) > 0.3

def extract_blocking_keys(norm_name: str, norm_address: str, country: str):
    """
    Extracts blocking keys for dual name + address blocking.
    """
    keys = set()
    ctry = country.upper().strip() if country else "UNK"
    is_non_lat = is_non_latin_text(norm_name)
    
    # 1. Name Keys (for Latin script names)
    if not is_non_lat and norm_name:
        clean_n = RE_CLEAN.sub(' ', norm_name).lower()
        tokens = [t for t in clean_n.split() if len(t) >= 3 and t not in NAME_STOPWORDS]
        for t in tokens[:2]:
            if len(t) >= 5:
                keys.add(f"N1:{ctry}:{t}")
        if len(tokens) >= 2:
            keys.add(f"N2:{ctry}:{tokens[0]}_{tokens[1]}")
            keys.add(f"NP:{ctry}:{tokens[0][:4]}_{tokens[1][:4]}")

    # 2. Address Keys (For ALL entities, mandatory for Non-Latin matching)
    if norm_address:
        pincodes = RE_PINCODE.findall(norm_address)
        for pin in pincodes:
            keys.add(f"PIN:{ctry}:{pin}")

        clean_a = RE_CLEAN.sub(' ', norm_address).lower()
        tokens = [t for t in clean_a.split() if t not in ADDR_STOPWORDS]
        digits = [t for t in tokens if t.isdigit() and 1 <= len(t) <= 6]
        words = [t for t in tokens if t.isalpha() and len(t) >= 3]
        
        if digits and words:
            keys.add(f"A_NW:{ctry}:{digits[0]}_{words[0]}")
            
        if len(words) >= 2:
            top_words = words[:5]
            for i in range(len(top_words)):
                for j in range(i+1, len(top_words)):
                    w1, w2 = sorted([top_words[i], top_words[j]])
                    keys.add(f"A_W2:{ctry}:{w1}_{w2}")
        elif len(words) == 1 and len(words[0]) >= 5:
            keys.add(f"A_W1:{ctry}:{words[0]}")
            
    return keys

def build_inverted_index(records, max_block_size: int = 3000):
    """
    Builds an inverted index mapping block_key -> list of candidate entity IDs.
    records: list of tuples (entity_id, norm_name, norm_address, country)
    """
    index = defaultdict(list)
    for entity_id, bname, addr, country in records:
        keys = extract_blocking_keys(bname, addr, country)
        for k in keys:
            index[k].append(entity_id)
            
    # Prune mega-blocks exceeding max_block_size
    pruned_index = {k: v for k, v in index.items() if len(v) <= max_block_size}
    return pruned_index

def generate_candidates_for_entity(norm_name, norm_address, country, inverted_index, max_cands: int = 250):
    """
    Generates a candidate set of S2/S3 entity IDs for a given Source 1 entity.
    """
    keys = extract_blocking_keys(norm_name, norm_address, country)
    cand_set = set()
    for k in keys:
        if k in inverted_index:
            cand_set.update(inverted_index[k])
            if len(cand_set) >= max_cands:
                break
    return list(cand_set)



import os
import sqlite3

class DiskInvertedIndex:
    def __init__(self, db_path: str = "blocking_index.db", max_block_size: int = 3000):
        self.db_path = db_path
        self.max_block_size = max_block_size
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
            
        self.conn = sqlite3.connect(self.db_path)
        self.conn.execute("PRAGMA synchronous = OFF")
        self.conn.execute("PRAGMA journal_mode = MEMORY")
        self.conn.execute("DROP TABLE IF EXISTS index_table")
        self.conn.execute("DROP TABLE IF EXISTS records")
        self.conn.execute("CREATE TABLE records (entity_id TEXT PRIMARY KEY, raw_name TEXT, norm_name TEXT, norm_address TEXT, country TEXT, is_non_latin INTEGER)")
        self.conn.execute("CREATE TABLE index_table (block_key TEXT, entity_id TEXT)")
        
    def build(self, records, batch_size: int = 100000):
        """
        records: sequence of tuples (entity_id, raw_name, norm_name, norm_address, country, is_non_latin)
        """
        cursor = self.conn.cursor()
        rec_buffer = []
        idx_buffer = []
        
        for item in records:
            eid, raw_n, norm_n, addr, country, nl_flag = item[0], item[1], item[2], item[3], item[4], item[5]
            eid_str = str(eid)
            raw_n_str = str(raw_n) if raw_n is not None else ""
            norm_n_str = str(norm_n) if norm_n is not None else ""
            addr_str = str(addr) if addr is not None else ""
            country_str = str(country) if country is not None else ""
            nl_int = 1 if nl_flag else 0
            
            rec_buffer.append((eid_str, raw_n_str, norm_n_str, addr_str, country_str, nl_int))
            
            keys = extract_blocking_keys(norm_n_str, addr_str, country_str)
            for k in keys:
                idx_buffer.append((k, eid_str))
                
            if len(rec_buffer) >= batch_size:
                cursor.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?, ?)", rec_buffer)
                cursor.executemany("INSERT INTO index_table VALUES (?, ?)", idx_buffer)
                self.conn.commit()
                rec_buffer.clear()
                idx_buffer.clear()
                
        if rec_buffer:
            cursor.executemany("INSERT OR REPLACE INTO records VALUES (?, ?, ?, ?, ?, ?)", rec_buffer)
            cursor.executemany("INSERT INTO index_table VALUES (?, ?)", idx_buffer)
            self.conn.commit()
            rec_buffer.clear()
            idx_buffer.clear()
            
        # Index finalized separately via finalize_index()
        pass

    def finalize_index(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON index_table(block_key)")
        self.conn.commit()

    def get_records_by_ids(self, entity_ids: list) -> dict:
        if not entity_ids:
            return {}
        cursor = self.conn.cursor()
        result = {}
        # Fetch in chunks of 900 to stay within SQLite parameter limits
        chunk_size = 900
        for i in range(0, len(entity_ids), chunk_size):
            chunk = [str(x) for x in entity_ids[i:i+chunk_size]]
            placeholders = ','.join(['?'] * len(chunk))
            cursor.execute(f"SELECT entity_id, raw_name, norm_name, norm_address, country, is_non_latin FROM records WHERE entity_id IN ({placeholders})", chunk)
            for r in cursor.fetchall():
                result[r[0]] = {
                    "business_name": r[1],
                    "norm_name": r[2],
                    "norm_address": r[3],
                    "country": r[4],
                    "is_non_latin": bool(r[5])
                }
        return result

    def generate_candidates(self, norm_name: str, norm_address: str, country: str, max_cands: int = 250) -> list:
        keys = extract_blocking_keys(norm_name, norm_address, country)
        if not keys:
            return []
            
        cursor = self.conn.cursor()
        cand_set = set()
        
        for k in keys:
            cursor.execute(
                "SELECT entity_id FROM index_table WHERE block_key = ? LIMIT ?",
                (k, self.max_block_size + 1)
            )
            rows = cursor.fetchall()
            if len(rows) <= self.max_block_size:
                for row in rows:
                    cand_set.add(row[0])
                    if len(cand_set) >= max_cands:
                        break
            if len(cand_set) >= max_cands:
                break
                
        return list(cand_set)

    def close(self):
        if self.conn:
            self.conn.close()
            
    def cleanup(self):
        self.close()
        if os.path.exists(self.db_path):
            try:
                os.remove(self.db_path)
            except Exception:
                pass
