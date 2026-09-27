with open('code/business_entity_resolution/src/blocking.py', 'r', encoding='utf-8') as f:
    text = f.read()

text = text.replace("CREATE INDEX idx_key ON index_table(block_key)", "CREATE INDEX IF NOT EXISTS idx_key ON index_table(block_key)")

with open('code/business_entity_resolution/src/blocking.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('blocking.py index fix applied successfully!')