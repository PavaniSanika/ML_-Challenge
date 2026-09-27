import os, re, sqlite3

# 1. Update features.py
with open('code/business_entity_resolution/src/features.py', 'r', encoding='utf-8') as f:
    feat_text = f.read()

new_func = '''def get_script_type(text: str) -> str:
    if not text or not text.strip():
        return "empty"
        
    devanagari = sum(1 for ch in text if 0x0900 <= ord(ch) <= 0x097F)
    bengali = sum(1 for ch in text if 0x0980 <= ord(ch) <= 0x09FF)
    gurmukhi = sum(1 for ch in text if 0x0A00 <= ord(ch) <= 0x0A7F)
    gujarati = sum(1 for ch in text if 0x0A80 <= ord(ch) <= 0x0AFF)
    odia = sum(1 for ch in text if 0x0B00 <= ord(ch) <= 0x0B7F)
    tamil = sum(1 for ch in text if 0x0B80 <= ord(ch) <= 0x0BFF)
    telugu = sum(1 for ch in text if 0x0C00 <= ord(ch) <= 0x0C7F)
    kannada = sum(1 for ch in text if 0x0C80 <= ord(ch) <= 0x0CFF)
    malayalam = sum(1 for ch in text if 0x0D00 <= ord(ch) <= 0x0D7F)
    cyrillic = sum(1 for ch in text if 0x0400 <= ord(ch) <= 0x04FF)
    arabic = sum(1 for ch in text if 0x0600 <= ord(ch) <= 0x06FF)
    cjk = sum(1 for ch in text if 0x4E00 <= ord(ch) <= 0x9FFF)
    hangul = sum(1 for ch in text if 0xAC00 <= ord(ch) <= 0xD7AF)
    thai = sum(1 for ch in text if 0x0E00 <= ord(ch) <= 0x0E7F)
    latin = sum(1 for ch in text if ord(ch) <= 0x024F and ch.isalpha())
    
    counts = {
        "devanagari": devanagari,
        "bengali": bengali,
        "gurmukhi": gurmukhi,
        "gujarati": gujarati,
        "odia": odia,
        "tamil": tamil,
        "telugu": telugu,
        "kannada": kannada,
        "malayalam": malayalam,
        "cyrillic": cyrillic,
        "arabic": arabic,
        "cjk": cjk,
        "hangul": hangul,
        "thai": thai,
        "latin": latin
    }
    max_script = max(counts, key=counts.get)
    if counts[max_script] == 0:
        return "unknown"
    return max_script'''

pattern = r'def get_script_type\(text: str\) -> str:.*?(?=\n\ndef char_ngrams)'
feat_text = re.sub(pattern, new_func, feat_text, flags=re.DOTALL)

with open('code/business_entity_resolution/src/features.py', 'w', encoding='utf-8') as f:
    f.write(feat_text)

print("FEATURES_UPDATED_SUCCESSFULLY")

# 2. Update blocking.py
with open('code/business_entity_resolution/src/blocking.py', 'r', encoding='utf-8') as f:
    block_text = f.read()

if 'DROP TABLE IF EXISTS index_table' not in block_text:
    old_init = '        self.conn.execute("CREATE TABLE index_table (block_key TEXT, entity_id TEXT)")'
    new_init = '        self.conn.execute("DROP TABLE IF EXISTS index_table")\n        self.conn.execute("CREATE TABLE index_table (block_key TEXT, entity_id TEXT)")'
    block_text = block_text.replace(old_init, new_init)
    with open('code/business_entity_resolution/src/blocking.py', 'w', encoding='utf-8') as f:
        f.write(block_text)

print("BLOCKING_UPDATED_SUCCESSFULLY")