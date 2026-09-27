"""
Data Normalization Pipeline for Business Entity Resolution.
Covers:
1. Mojibake repair (e.g. Â€“ -> -, â€™ -> ')
2. Non-Latin script detection (% non-Latin codepoints > 30%)
3. Null & OCR ## cleanup in addresses (literal 'NULL' -> '', ##2326 -> 2326)
4. Domain name & markdown link removal
5. Trailing reference number stripping (- 2067865001)
6. DBA trade name extraction (mid-string dba)
7. Country-scoped legal suffix stripping (LLC, Inc, Corp, Pvt, Ltd, SARL, SASU, EURL, SAS, SA)
8. Accent folding (Unicode NFKD normalization)
9. Leetspeak & missing space boundary repair (5uperior -> superior, Doorno.52 -> doorno 52, 9Th -> 9 th)
10. Whitespace & non-alphanumeric punctuation cleaning
"""

import re
import unicodedata
import pandas as pd

MOJIBAKE_MAP = {
    "â€™": "'",
    "â€˜": "'",
    "â€œ": '"',
    "â€ ": '"',
    "Â€“": "-",
    "â€“": "-",
    "â€”": "-",
    "Â ": " ",
    "Â": "",
    "Ã©": "é",
    "Ã¨": "è",
    "Ã ": "à",
    "Ã¢": "â",
    "Ã®": "î",
    "Ã¯": "ï",
    "Ã´": "ô",
    "Ã»": "û",
    "Ã§": "ç",
    "Ã‰": "É",
    "Ãˆ": "È",
    "Ã€": "À",
    "Â°": "",
    "â€¢": " ",
    "â€¦": " "
}

LEET_MAP = {'5': 's', '0': 'o', '1': 'i', '3': 'e', '8': 'b'}

# Pre-compiled Regex Patterns
RE_MOJIBAKE = re.compile("|".join(map(re.escape, MOJIBAKE_MAP.keys())))
RE_MARKDOWN = re.compile(r'\[(.*?)\]\(.*?\)', re.IGNORECASE)
RE_URL = re.compile(r'https?://\S+', re.IGNORECASE)
RE_DOMAIN = re.compile(r'\b([a-zA-Z0-9-]+)\.(com|org|net|in|fr|co|io|biz|info)\b', re.IGNORECASE)
RE_TRAILING_REF = re.compile(r'-\s*\d{5,}\s*$', re.IGNORECASE)
RE_LEADING_HASH = re.compile(r'##(\d+)', re.IGNORECASE)
RE_ORDINALS = re.compile(r'(\d+)(st|nd|rd|th)([a-zA-Z])', re.IGNORECASE)
RE_LETTER_PUNCT_DIGIT = re.compile(r'([a-zA-Z]+)[\.\,\#\/](\d+)', re.IGNORECASE)
RE_DBA_MID = re.compile(r'(?<=\S\s)\b(dba|d/b/a)\b(?=\s\S)', re.IGNORECASE)
RE_DBA_START = re.compile(r'^(dba|d/b/a)\s+', re.IGNORECASE)

# Legal Suffixes
RE_SUFFIX_END_LONG = re.compile(r'\b(Corporation|Private|Limited|Incorporated|Company|Corp|Pvt|Ltd|LLC|Inc|LLP|Co|SARL|SASU|EURL)\b$', re.IGNORECASE)
RE_SUFFIX_START_LONG = re.compile(r'^(LLC|Inc|Corp|Pvt|Ltd|LLP|SARL|SASU|EURL)\b', re.IGNORECASE)
RE_COMPANY_END = re.compile(r'\b(Company|Co)\b$', re.IGNORECASE)

RE_SA_SAS_END = re.compile(r'\b(SAS|SA)$', re.IGNORECASE)
RE_SA_SAS_START_FR = re.compile(r'^(SAS|SA)\b', re.IGNORECASE)

def fix_mojibake(text: str) -> str:
    if not text:
        return ""
    if any(m in text for m in ["Â", "â€", "Ã"]):
        return RE_MOJIBAKE.sub(lambda m: MOJIBAKE_MAP[m.group(0)], text)
    return text

def fold_accents(text: str) -> str:
    if not text:
        return ""
    nfkd = unicodedata.normalize('NFKD', text)
    return "".join([c for c in nfkd if not unicodedata.combining(c)])

def fix_leetspeak_and_boundaries(text: str) -> str:
    if not text:
        return ""
    text = RE_ORDINALS.sub(r'\1 \2 \3', text)
    text = RE_LETTER_PUNCT_DIGIT.sub(r'\1 \2', text)
    
    words = text.split()
    new_words = []
    for w in words:
        if len(w) >= 4 and w[0] in LEET_MAP and w[1:].isalpha():
            new_words.append(LEET_MAP[w[0]] + w[1:])
        else:
            w_split = re.sub(r'([a-zA-Z]+)(\d+)', r'\1 \2', w)
            w_split = re.sub(r'(\d+)([a-zA-Z]+)', r'\1 \2', w_split)
            new_words.append(w_split)
            
    return " ".join(new_words)

def strip_suffixes(name: str, country: str) -> str:
    if not name:
        return ""
    name = name.strip()
    
    # Iteratively strip legal suffixes stacked at the end (e.g. "Acme Pvt Ltd" -> "Acme Pvt" -> "Acme")
    while True:
        new_name = RE_SUFFIX_END_LONG.sub('', name).strip()
        new_name = RE_COMPANY_END.sub('', new_name).strip()
        if country in ["France", "FR"]:
            new_name = RE_SA_SAS_END.sub('', new_name).strip()
        else:
            new_name = RE_SA_SAS_END.sub('', new_name).strip()
        if new_name == name:
            break
        name = new_name

    # Iteratively strip legal suffixes stacked at the start (e.g. "LLC Corp Safe Newbury")
    while True:
        new_name = RE_SUFFIX_START_LONG.sub('', name).strip()
        if country in ["France", "FR"]:
            new_name = RE_SA_SAS_START_FR.sub('', new_name).strip()
        if new_name == name:
            break
        name = new_name
        
    return name.strip()

def normalize_single_record(bname: str, addr: str, country: str = ""):
    if not bname:
        bname = ""
    if not addr:
        addr = ""
        
    # 1. Mojibake repair
    bname = fix_mojibake(bname)
    addr = fix_mojibake(addr)
    
    # 2. Check non-Latin script
    non_latin_chars = sum(1 for ch in bname if ord(ch) > 0x024F)
    is_non_latin = (non_latin_chars / max(1, len(bname))) > 0.3
    
    # 3. Handle NULL and OCR ## in address
    if addr.strip().upper() == "NULL":
        addr = ""
    else:
        addr = RE_LEADING_HASH.sub(r'\1', addr)
        
    # 4. Handle domain and markdown in name
    bname = RE_MARKDOWN.sub(r'\1', bname)
    bname = RE_URL.sub('', bname)
    bname = RE_DOMAIN.sub(r'\1', bname)
    
    # 5. Handle trailing ref numbers in name
    bname = RE_TRAILING_REF.sub('', bname)
    
    # 6. Handle dba trade names (strip leading dba or take left side of mid-string dba)
    bname = RE_DBA_START.sub('', bname).strip()
    dba_split = RE_DBA_MID.split(bname)
    if len(dba_split) > 1:
        bname = dba_split[0].strip()
        
    # 7. Suffix stripping (for Latin names)
    if not is_non_latin:
        bname = strip_suffixes(bname, country)
        
    # 8. Accent folding
    bname = fold_accents(bname)
    addr = fold_accents(addr)
    
    # 9. Leetspeak & boundary repair
    bname = fix_leetspeak_and_boundaries(bname)
    addr = fix_leetspeak_and_boundaries(addr)
    
    # 10. Clean non-alphanumeric punctuation & lower
    bname_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', bname).lower()
    bname_clean = re.sub(r'\s+', ' ', bname_clean).strip()
    
    addr_clean = re.sub(r'[^a-zA-Z0-9\s]', ' ', addr).lower()
    addr_clean = re.sub(r'\s+', ' ', addr_clean).strip()
    
    return bname_clean, addr_clean, is_non_latin

def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Vectorized / Batch normalization for DataFrames containing
    'business_name', 'business_address', and 'country'.
    """
    names = df['business_name'].fillna('')
    addrs = df['business_address'].fillna('')
    countries = df['country'].fillna('')
    
    norm_names = []
    norm_addrs = []
    is_non_latins = []
    
    for bname, addr, ctry in zip(names, addrs, countries):
        nb, na, nl = normalize_single_record(bname, addr, ctry)
        norm_names.append(nb)
        norm_addrs.append(na)
        is_non_latins.append(nl)
        
    df['norm_name'] = norm_names
    df['norm_address'] = norm_addrs
    df['is_non_latin'] = is_non_latins
    return df
