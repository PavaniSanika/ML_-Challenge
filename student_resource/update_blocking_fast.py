with open('code/business_entity_resolution/src/blocking.py', 'r', encoding='utf-8') as f:
    text = f.read()

# Replace build method to remove index creation inside build loop
text = text.replace('cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON index_table(block_key)")\n        self.conn.commit()', '# Index finalized separately via finalize_index()')

if "def finalize_index(self):" not in text:
    finalize_code = '''    def finalize_index(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON index_table(block_key)")
        self.conn.commit()'''
    text = text.replace("def get_records_by_ids", finalize_code + "\n\n    def get_records_by_ids")

with open('code/business_entity_resolution/src/blocking.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('blocking.py finalize_index updated successfully!')