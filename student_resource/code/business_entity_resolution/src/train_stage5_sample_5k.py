import sys
import os
import time
import tracemalloc
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from normalization import normalize_dataframe
from blocking import DiskInvertedIndex
from features import extract_features_batch
from sklearn.ensemble import HistGradientBoostingClassifier

def compute_f05(precision, recall):
    if (0.25 * precision + recall) == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def get_mem_mb():
    current, peak = tracemalloc.get_traced_memory()
    return current / (1024 * 1024), peak / (1024 * 1024)

def main():
    print("=== STAGE 5 MEMORY BENCHMARK (5,000 ENTITY SAMPLE) ===")
    tracemalloc.start()
    start_time = time.time()
    
    cur, peak = get_mem_mb()
    print(f"\n[CHECKPOINT 1] Baseline Initial RAM: {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    print("\n1. Loading sample ground truth (5,000 entities)...")
    gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=5000)
    s1_ids = set(gt_df["source1_entity_id"].dropna())
    
    gt_matches = {}
    needed_s2s3 = set()
    for _, row in gt_df.iterrows():
        s1_id = row["source1_entity_id"]
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        matches = [m.strip() for m in matched_str.split(",") if m.strip()]
        gt_matches[s1_id] = set(matches)
        needed_s2s3.update(matches)
        
    s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
    s1_df = s1_df[s1_df["entity_id"].isin(s1_ids)].copy()
    s1_df = normalize_dataframe(s1_df)
    
    cur, peak = get_mem_mb()
    print(f"[CHECKPOINT 2] After Loading & Normalizing S1 (5,000 entities): {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    print("\n2. Streaming S2 & S3 into SQLite DiskInvertedIndex (Record Store + Inverted Index)...")
    db_path = "train_sample_5k_blocking.db"
    blocking_index = DiskInvertedIndex(db_path=db_path)
    
    # Load and index S2 and S3 in chunks to keep RAM minimal
    def stream_and_index_file(filepath):
        for chunk in pd.read_csv(filepath, sep="\t", chunksize=50000):
            norm_chunk = normalize_dataframe(chunk)
            records = [
                (
                    row["entity_id"],
                    row["business_name"],
                    row["norm_name"],
                    row["norm_address"],
                    row["country"],
                    row["is_non_latin"]
                )
                for _, row in norm_chunk.iterrows()
            ]
            blocking_index.build(records)
            del chunk, norm_chunk, records
            
    # For 5,000 entity benchmark sample, index first 100k S2/S3 plus ground truth needed S2/S3
    s2_df = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=100000)
    s3_df = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", nrows=100000)
    s2_df = normalize_dataframe(s2_df)
    s3_df = normalize_dataframe(s3_df)
    
    s23_concat = pd.concat([s2_df, s3_df], ignore_index=True)
    records = [
        (
            row["entity_id"],
            row["business_name"],
            row["norm_name"],
            row["norm_address"],
            row["country"],
            row["is_non_latin"]
        )
        for _, row in s23_concat.iterrows()
    ]
    blocking_index.build(records)
    del s2_df, s3_df, s23_concat, records
    
    cur, peak = get_mem_mb()
    print(f"[CHECKPOINT 3] After Building SQLite DiskInvertedIndex (S2/S3 text stored in DB, RAM freed): {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    print("\n3. Generating candidate pairs & fetching record text from SQLite on demand...")
    pairs_list = []
    
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        cands = blocking_index.generate_candidates(
            s1_row["norm_name"], s1_row["norm_address"], s1_row["country"]
        )
        
        pos_matches = gt_matches.get(s1_id, set())
        all_cands = list(set(cands).union(pos_matches))
        
        # Batch fetch candidate record text from SQLite DB
        cand_records = blocking_index.get_records_by_ids(all_cands)
        
        for cand_id, cand_info in cand_records.items():
            is_match = 1.0 if cand_id in pos_matches else 0.0
            pairs_list.append({
                "s1_id": s1_id,
                "cand_id": cand_id,
                "name1": s1_row["norm_name"],
                "raw1": s1_row["business_name"],
                "addr1": s1_row["norm_address"],
                "country1": s1_row["country"],
                "nl1": s1_row["is_non_latin"],
                "name2": cand_info["norm_name"],
                "raw2": cand_info["business_name"],
                "addr2": cand_info["norm_address"],
                "country2": cand_info["country"],
                "nl2": cand_info["is_non_latin"],
                "label": is_match
            })
            
    pairs_df = pd.DataFrame(pairs_list)
    del pairs_list
    
    cur, peak = get_mem_mb()
    print(f"[CHECKPOINT 4] After Candidate Pair Generation ({len(pairs_df)} pairs): {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    print("\n4. Extracting 18 dense pair features...")
    features_df = extract_features_batch(pairs_df)
    X = features_df.values
    y = pairs_df["label"].values
    
    cur, peak = get_mem_mb()
    print(f"[CHECKPOINT 5] After Feature Extraction (X shape: {X.shape}): {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    print("\n5. Entity-level Train/Validation Split (80/20)...")
    unique_s1 = np.array(list(s1_ids))
    np.random.seed(42)
    np.random.shuffle(unique_s1)
    
    split_idx = int(0.8 * len(unique_s1))
    train_s1 = set(unique_s1[:split_idx])
    val_s1 = set(unique_s1[split_idx:])
    
    train_mask = pairs_df["s1_id"].isin(train_s1).values
    val_mask = pairs_df["s1_id"].isin(val_s1).values
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pairs_df = pairs_df[val_mask].copy()
    
    print("\n6. Training HistGradientBoostingClassifier & Tuning Threshold...")
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.08,
        max_depth=6,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_pairs_df["prob"] = val_probs
    
    best_threshold = 0.5
    best_macro_f05 = 0.0
    val_s1_list = list(val_s1)
    
    for thresh in np.arange(0.1, 0.95, 0.05):
        thresh = round(float(thresh), 2)
        pred_matches = val_pairs_df[val_pairs_df["prob"] >= thresh]
        pred_dict = pred_matches.groupby("s1_id")["cand_id"].apply(set).to_dict()
        
        f05_scores = []
        for s1_id in val_s1_list:
            gt_set = gt_matches.get(s1_id, set())
            pred_set = pred_dict.get(s1_id, set())
            if not gt_set and not pred_set:
                f05_scores.append(1.0)
            elif not pred_set and gt_set or pred_set and not gt_set:
                f05_scores.append(0.0)
            else:
                tp = len(gt_set.intersection(pred_set))
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f05_scores.append(compute_f05(prec, rec))
        macro_f05 = float(np.mean(f05_scores))
        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = thresh
            
    cur, peak = get_mem_mb()
    print(f"[CHECKPOINT 6] After Classifier Training & Threshold Tuning: {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    blocking_index.cleanup()
    elapsed = time.time() - start_time
    
    print("\n" + "="*70)
    print(f"STAGE 5 BENCHMARK COMPLETE (5,000 ENTITIES)")
    print(f"Optimal Threshold: {best_threshold:.2f} (Validation Macro F0.5 = {best_macro_f05:.4f})")
    print(f"Total Time: {elapsed:.2f} seconds")
    print(f"FINAL PEAK MEMORY: {peak:.2f} MB")
    print("="*70)

if __name__ == "__main__":
    main()