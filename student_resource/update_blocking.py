import re, os, sqlite3

with open('code/business_entity_resolution/src/blocking.py', 'r', encoding='utf-8') as f:
    text = f.read()

disk_class_code = '''class DiskInvertedIndex:
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
            
        cursor.execute("CREATE INDEX idx_key ON index_table(block_key)")
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
                pass'''

pattern = r'class DiskInvertedIndex:.*?(?=$)'
text = re.sub(pattern, disk_class_code, text, flags=re.DOTALL)

with open('code/business_entity_resolution/src/blocking.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('blocking.py updated with SQLite Record Store successfully!')