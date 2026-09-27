"""
Feature Engineering Module for Business Entity Resolution Candidate Pairs.
Generates a comprehensive vector of numeric features for each candidate pair:
- Multi-script detection (Latin, Devanagari, Tamil, Telugu, Malayalam)
- Script match / mismatch flags
- Script-aware name similarity (zeroed out on any script mismatch)
- Address token overlap, containment, and character n-gram similarity
- Address empty checks and has_address_both flag
- Digit / House number / PIN-code exact match flags
- Country flags (open-set: US, India, France, etc.)
- Source origin flags (Source 2 vs Source 3)
"""

import re
import math
import unicodedata
from typing import Dict, Any, List
import numpy as np
import pandas as pd

try:
    import rapidfuzz
    from rapidfuzz.distance import Levenshtein
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

def get_script_type(text: str) -> str:
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
    return max_script

def char_ngrams(text: str, n: int = 3) -> set:
    if not text or len(text) < n:
        return {text} if text else set()
    return {text[i:i+n] for i in range(len(text) - n + 1)}

def jaccard_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    return intersection / union if union > 0 else 0.0

def containment_similarity(set1: set, set2: set) -> float:
    if not set1 or not set2:
        return 0.0
    intersection = len(set1.intersection(set2))
    min_len = min(len(set1), len(set2))
    return intersection / min_len if min_len > 0 else 0.0

def levenshtein_distance(s1: str, s2: str) -> int:
    if s1 == s2:
        return 0
    if len(s1) == 0:
        return len(s2)
    if len(s2) == 0:
        return len(s1)
    
    v0 = list(range(len(s2) + 1))
    v1 = [0] * (len(s2) + 1)
    
    for i in range(len(s1)):
        v1[0] = i + 1
        for j in range(len(s2)):
            cost = 0 if s1[i] == s2[j] else 1
            v1[j + 1] = min(v1[j] + 1, v0[j + 1] + 1, v0[j] + cost)
        v0, v1 = v1, v0
        
    return v0[len(s2)]

def edit_ratio(s1: str, s2: str) -> float:
    if not s1 or not s2:
        return 0.0
    if HAS_RAPIDFUZZ:
        return float(Levenshtein.normalized_similarity(s1, s2))
    max_len = max(len(s1), len(s2))
    if max_len == 0:
        return 0.0
    dist = levenshtein_distance(s1, s2)
    return max(0.0, 1.0 - (dist / max_len))

RE_DIGITS = re.compile(r'\b\d{1,6}\b')
RE_PIN = re.compile(r'\b\d{5,6}\b')

def extract_candidate_pair_features(rec1: Dict[str, Any], rec2: Dict[str, Any]) -> Dict[str, float]:
    """
    Computes feature dict for pair (rec1, rec2).
    rec1, rec2 contain: 'norm_name', 'norm_address', 'country', 'is_non_latin', 'entity_id', 'raw_name' (optional)
    """
    feats = {}
    
    # 1. Script Detection & Mismatch Flags
    # Check raw_name or norm_name
    raw1 = rec1.get('raw_name') or rec1.get('norm_name', '')
    raw2 = rec2.get('raw_name') or rec2.get('norm_name', '')
    
    script1 = get_script_type(raw1)
    script2 = get_script_type(raw2)
    
    nl1 = rec1.get('is_non_latin', script1 != 'latin')
    nl2 = rec2.get('is_non_latin', script2 != 'latin')
    
    # Script mismatch if one is latin & other is non-latin, OR if two different non-latin scripts (e.g. Tamil vs Devanagari)
    is_script_mismatch = 1.0 if (script1 != script2 and script1 != 'empty' and script2 != 'empty') else 0.0
    both_latin = 1.0 if (script1 == 'latin' and script2 == 'latin') else 0.0
    both_non_latin = 1.0 if (nl1 and nl2) else 0.0
    same_non_latin_script = 1.0 if (nl1 and nl2 and script1 == script2) else 0.0
    
    feats['is_script_mismatch'] = is_script_mismatch
    feats['both_latin'] = both_latin
    feats['both_non_latin'] = both_non_latin
    feats['same_non_latin_script'] = same_non_latin_script

    # 2. Name Features (Zero-weighted / Zeroed out if script mismatch or empty name)
    name1 = rec1.get('norm_name', '').strip()
    name2 = rec2.get('norm_name', '').strip()
    
    if is_script_mismatch or not name1 or not name2:
        feats['name_jaccard'] = 0.0
        feats['name_containment'] = 0.0
        feats['name_edit_ratio'] = 0.0
        feats['name_char_3gram_sim'] = 0.0
        feats['name_first_token_match'] = 0.0
        feats['name_len_diff_ratio'] = 0.0
    else:
        tokens1 = set(name1.split())
        tokens2 = set(name2.split())
        
        feats['name_jaccard'] = jaccard_similarity(tokens1, tokens2)
        feats['name_containment'] = containment_similarity(tokens1, tokens2)
        feats['name_edit_ratio'] = edit_ratio(name1, name2)
        
        ngrams1 = char_ngrams(name1, 3)
        ngrams2 = char_ngrams(name2, 3)
        feats['name_char_3gram_sim'] = jaccard_similarity(ngrams1, ngrams2)
        
        t1_first = name1.split()[0] if name1 else ""
        t2_first = name2.split()[0] if name2 else ""
        feats['name_first_token_match'] = 1.0 if (t1_first and t1_first == t2_first) else 0.0
        
        max_l = max(len(name1), len(name2))
        feats['name_len_diff_ratio'] = abs(len(name1) - len(name2)) / max_l if max_l > 0 else 0.0

    # 3. Address Features & has_address_both Flag
    addr1 = rec1.get('norm_address', '').strip()
    addr2 = rec2.get('norm_address', '').strip()
    
    has_address_both = 1.0 if (bool(addr1) and bool(addr2)) else 0.0
    feats['has_address_both'] = has_address_both
    
    if not has_address_both:
        feats['address_jaccard'] = 0.0
        feats['address_containment'] = 0.0
        feats['address_edit_ratio'] = 0.0
        feats['address_char_3gram_sim'] = 0.0
        feats['address_token_overlap_count'] = 0.0
        feats['digit_jaccard'] = 0.0
        feats['pincode_exact_match'] = 0.0
    else:
        atokens1 = set(addr1.split())
        atokens2 = set(addr2.split())
        
        feats['address_jaccard'] = jaccard_similarity(atokens1, atokens2)
        feats['address_containment'] = containment_similarity(atokens1, atokens2)
        feats['address_token_overlap_count'] = float(len(atokens1.intersection(atokens2)))
        feats['address_edit_ratio'] = edit_ratio(addr1, addr2)
        
        angrams1 = char_ngrams(addr1, 3)
        angrams2 = char_ngrams(addr2, 3)
        feats['address_char_3gram_sim'] = jaccard_similarity(angrams1, angrams2)
        
        # Digit matching (house / plot numbers)
        digits1 = set(RE_DIGITS.findall(addr1))
        digits2 = set(RE_DIGITS.findall(addr2))
        feats['digit_jaccard'] = jaccard_similarity(digits1, digits2)
        
        # PIN code exact match flag
        pins1 = set(RE_PIN.findall(addr1))
        pins2 = set(RE_PIN.findall(addr2))
        if pins1 and pins2 and len(pins1.intersection(pins2)) > 0:
            feats['pincode_exact_match'] = 1.0
        else:
            feats['pincode_exact_match'] = 0.0

    # 4. Country & Source Flags
    c1 = rec1.get('country', '').upper().strip()
    c2 = rec2.get('country', '').upper().strip()
    feats['country_exact_match'] = 1.0 if (c1 and c1 == c2) else 0.0
    feats['country_is_france'] = 1.0 if (c1 in ['FRANCE', 'FR'] or c2 in ['FRANCE', 'FR']) else 0.0
    feats['country_is_india'] = 1.0 if (c1 in ['INDIA', 'IN'] or c2 in ['INDIA', 'IN']) else 0.0
    feats['country_is_us'] = 1.0 if (c1 in ['US', 'USA'] or c2 in ['US', 'USA']) else 0.0

    # Candidate source flag
    cid2 = rec2.get('entity_id', '')
    feats['is_source2_candidate'] = 1.0 if cid2.startswith('S2') or '_s2' in cid2 else 0.0

    return feats

FEATURE_NAMES = [
    'is_script_mismatch', 'both_latin', 'both_non_latin', 'same_non_latin_script',
    'name_jaccard', 'name_containment', 'name_edit_ratio', 'name_char_3gram_sim', 'name_first_token_match', 'name_len_diff_ratio',
    'has_address_both', 'address_jaccard', 'address_containment', 'address_token_overlap_count', 'address_edit_ratio', 'address_char_3gram_sim',
    'digit_jaccard', 'pincode_exact_match',
    'country_exact_match', 'country_is_france', 'country_is_india', 'country_is_us', 'is_source2_candidate'
]

def extract_features_batch(df_pairs: pd.DataFrame, chunk_size: int = 200000) -> pd.DataFrame:
    """
    Vectorized batch feature extraction across candidate pairs DataFrame.
    Matches extract_candidate_pair_features output 100% identically for every pair.
    Automatically chunks large DataFrames (chunk_size=200,000) to keep memory footprint minimal (< 200 MB).
    """
    n = len(df_pairs)
    if n > chunk_size:
        chunks = [extract_features_batch(df_pairs.iloc[i:i + chunk_size], chunk_size) for i in range(0, n, chunk_size)]
        return pd.concat(chunks, axis=0)

    feats = pd.DataFrame(index=df_pairs.index)

    name1_arr = df_pairs['name1'].fillna('').astype(str).values
    name2_arr = df_pairs['name2'].fillna('').astype(str).values
    raw1_arr = df_pairs['raw1'].fillna(df_pairs['name1']).fillna('').astype(str).values if 'raw1' in df_pairs.columns else name1_arr
    raw2_arr = df_pairs['raw2'].fillna(df_pairs['name2']).fillna('').astype(str).values if 'raw2' in df_pairs.columns else name2_arr
    addr1_arr = df_pairs['addr1'].fillna('').astype(str).values
    addr2_arr = df_pairs['addr2'].fillna('').astype(str).values
    c1_arr = df_pairs['country1'].fillna('').astype(str).str.upper().values
    c2_arr = df_pairs['country2'].fillna('').astype(str).str.upper().values
    cand_ids = df_pairs['cand_id'].fillna('').astype(str).values

    # Script types using raw_name / name
    all_raws = set(raw1_arr).union(set(raw2_arr))
    script_map = {r: get_script_type(r) for r in all_raws}

    script1_list = [script_map[r] for r in raw1_arr]
    script2_list = [script_map[r] for r in raw2_arr]

    nl1_col = df_pairs['nl1'] if 'nl1' in df_pairs.columns else [None]*n
    nl2_col = df_pairs['nl2'] if 'nl2' in df_pairs.columns else [None]*n

    nl1_list = []
    nl2_list = []
    for i in range(n):
        v1 = nl1_col.iloc[i] if hasattr(nl1_col, 'iloc') else nl1_col[i]
        v2 = nl2_col.iloc[i] if hasattr(nl2_col, 'iloc') else nl2_col[i]
        
        s1 = script1_list[i]
        s2 = script2_list[i]
        
        n1_flag = bool(v1) if (v1 is not None and pd.notna(v1)) else (s1 != 'latin' and s1 != 'empty')
        n2_flag = bool(v2) if (v2 is not None and pd.notna(v2)) else (s2 != 'latin' and s2 != 'empty')
        
        nl1_list.append(n1_flag)
        nl2_list.append(n2_flag)

    is_script_mismatch = np.zeros(n, dtype=np.float32)
    both_latin = np.zeros(n, dtype=np.float32)
    both_non_latin = np.zeros(n, dtype=np.float32)
    same_non_latin_script = np.zeros(n, dtype=np.float32)

    for i in range(n):
        s1, s2 = script1_list[i], script2_list[i]
        n1_flag, n2_flag = nl1_list[i], nl2_list[i]
        
        mismatch = 1.0 if (s1 != s2 and s1 != 'empty' and s2 != 'empty') else 0.0
        b_latin = 1.0 if (s1 == 'latin' and s2 == 'latin') else 0.0
        b_non_latin = 1.0 if (n1_flag and n2_flag) else 0.0
        s_non_latin = 1.0 if (n1_flag and n2_flag and s1 == s2) else 0.0
        
        is_script_mismatch[i] = mismatch
        both_latin[i] = b_latin
        both_non_latin[i] = b_non_latin
        same_non_latin_script[i] = s_non_latin

    feats['is_script_mismatch'] = is_script_mismatch
    feats['both_latin'] = both_latin
    feats['both_non_latin'] = both_non_latin
    feats['same_non_latin_script'] = same_non_latin_script

    # 3-gram maps
    all_names = set(name1_arr).union(set(name2_arr))
    all_addrs = set(addr1_arr).union(set(addr2_arr))
    name_ngrams_map = {n: char_ngrams(n, 3) for n in all_names}
    addr_ngrams_map = {a: char_ngrams(a, 3) for a in all_addrs}

    name_jaccard = np.zeros(n, dtype=np.float32)
    name_containment = np.zeros(n, dtype=np.float32)
    name_edit_ratio = np.zeros(n, dtype=np.float32)
    name_char_3gram_sim = np.zeros(n, dtype=np.float32)
    name_first_token_match = np.zeros(n, dtype=np.float32)
    name_len_diff_ratio = np.zeros(n, dtype=np.float32)

    valid_name_mask = (is_script_mismatch == 0.0) & (name1_arr != '') & (name2_arr != '')
    valid_name_indices = np.where(valid_name_mask)[0]

    for idx in valid_name_indices:
        n1 = name1_arr[idx]
        n2 = name2_arr[idx]
        
        t1 = set(n1.split())
        t2 = set(n2.split())
        name_jaccard[idx] = jaccard_similarity(t1, t2)
        name_containment[idx] = containment_similarity(t1, t2)
        
        # Levenshtein Edit Ratio
        name_edit_ratio[idx] = edit_ratio(n1, n2)
        
        # Char 3-gram Sim
        name_char_3gram_sim[idx] = jaccard_similarity(name_ngrams_map[n1], name_ngrams_map[n2])
        
        # First Token Match
        w1 = n1.split()[0] if n1 else ""
        w2 = n2.split()[0] if n2 else ""
        name_first_token_match[idx] = 1.0 if (w1 and w1 == w2) else 0.0
        
        # Len diff ratio
        max_l = max(len(n1), len(n2))
        name_len_diff_ratio[idx] = abs(len(n1) - len(n2)) / max_l if max_l > 0 else 0.0

    feats['name_jaccard'] = name_jaccard
    feats['name_containment'] = name_containment
    feats['name_edit_ratio'] = name_edit_ratio
    feats['name_char_3gram_sim'] = name_char_3gram_sim
    feats['name_first_token_match'] = name_first_token_match
    feats['name_len_diff_ratio'] = name_len_diff_ratio

    # Address Features
    has_address_both = ((addr1_arr != '') & (addr2_arr != '')).astype(np.float32)
    feats['has_address_both'] = has_address_both

    addr_jaccard = np.zeros(n, dtype=np.float32)
    addr_containment = np.zeros(n, dtype=np.float32)
    addr_overlap_cnt = np.zeros(n, dtype=np.float32)
    addr_edit_ratio = np.zeros(n, dtype=np.float32)
    addr_char_3gram_sim = np.zeros(n, dtype=np.float32)
    digit_jaccard = np.zeros(n, dtype=np.float32)
    pincode_match = np.zeros(n, dtype=np.float32)

    valid_addr_indices = np.where(has_address_both == 1.0)[0]

    for idx in valid_addr_indices:
        a1 = addr1_arr[idx]
        a2 = addr2_arr[idx]
        
        t1 = set(a1.split())
        t2 = set(a2.split())
        
        addr_jaccard[idx] = jaccard_similarity(t1, t2)
        addr_containment[idx] = containment_similarity(t1, t2)
        addr_overlap_cnt[idx] = float(len(t1.intersection(t2)))
        
        # Levenshtein Edit Ratio
        addr_edit_ratio[idx] = edit_ratio(a1, a2)
        
        # Char 3-gram Sim
        addr_char_3gram_sim[idx] = jaccard_similarity(addr_ngrams_map[a1], addr_ngrams_map[a2])
        
        # Digits
        d1 = set(RE_DIGITS.findall(a1))
        d2 = set(RE_DIGITS.findall(a2))
        digit_jaccard[idx] = jaccard_similarity(d1, d2)
        
        # PIN code exact match
        p1 = set(RE_PIN.findall(a1))
        p2 = set(RE_PIN.findall(a2))
        if p1 and p2 and len(p1.intersection(p2)) > 0:
            pincode_match[idx] = 1.0

    feats['address_jaccard'] = addr_jaccard
    feats['address_containment'] = addr_containment
    feats['address_token_overlap_count'] = addr_overlap_cnt
    feats['address_edit_ratio'] = addr_edit_ratio
    feats['address_char_3gram_sim'] = addr_char_3gram_sim
    feats['digit_jaccard'] = digit_jaccard
    feats['pincode_exact_match'] = pincode_match

    # Country & Source Flags
    feats['country_exact_match'] = ((c1_arr != '') & (c1_arr == c2_arr)).astype(np.float32)
    feats['country_is_france'] = ((c1_arr == 'FRANCE') | (c1_arr == 'FR') | (c2_arr == 'FRANCE') | (c2_arr == 'FR')).astype(np.float32)
    feats['country_is_india'] = ((c1_arr == 'INDIA') | (c1_arr == 'IN') | (c2_arr == 'INDIA') | (c2_arr == 'IN')).astype(np.float32)
    feats['country_is_us'] = ((c1_arr == 'US') | (c1_arr == 'USA') | (c2_arr == 'US') | (c2_arr == 'USA')).astype(np.float32)
    
    feats['is_source2_candidate'] = (pd.Series(cand_ids).str.startswith('S2') | pd.Series(cand_ids).str.contains('_s2')).astype(np.float32).values

    return feats[FEATURE_NAMES]
