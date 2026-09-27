with open('code/business_entity_resolution/src/blocking.py', 'r', encoding='utf-8') as f:
    text = f.read()

bad_block = '''        # Index finalized separately via finalize_index()

        def finalize_index(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON index_table(block_key)")
        self.conn.commit()'''

good_block = '''        # Index finalized separately via finalize_index()
        pass

    def finalize_index(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_key ON index_table(block_key)")
        self.conn.commit()'''

text = text.replace(bad_block, good_block)

with open('code/business_entity_resolution/src/blocking.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('blocking.py fixed successfully!')