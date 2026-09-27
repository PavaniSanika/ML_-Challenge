import sys
import os
import time
import tracemalloc
import pandas as pd
import numpy as np
import subprocess

sys.path.insert(0, os.path.dirname(__file__))

from normalization import normalize_dataframe
from blocking import DiskInvertedIndex
from features import extract_features_batch
from sklearn.ensemble import HistGradientBoostingClassifier

def compute_f05(precision, recall):
    if (0.25 * precision + recall) == 0:
        return 0.0
    return (1.25 * precision * recall) / (0.25 * precision + recall)

def write_valid_output(matching_dict, candidate_dict, s1_order, match_path, cand_path):
    """
    Writes 100% complete, valid matching_results.tsv and candidate_pairs.tsv.
    Guarantees every single required S1 ID is present (with predicted matches or empty string fallback).
    """
    with open(match_path, "w", encoding="utf-8") as fm:
        fm.write("source1_entity_id\tmatched_entity_ids\n")
        for s1_id in s1_order:
            fm.write(f"{s1_id}\t{matching_dict.get(s1_id, '')}\n")
            
    with open(cand_path, "w", encoding="utf-8") as fc:
        fc.write("source1_entity_id\tcandidate_entity_ids\n")
        for s1_id in s1_order:
            fc.write(f"{s1_id}\t{candidate_dict.get(s1_id, '')}\n")

def main():
    print("======================================================================")
    print("=== ULTRA-FAST DEADLINE PIPELINE WITH 100% SUBMISSION SAFETY NET ===")
    print("======================================================================")
    
    overall_start = time.time()
    tracemalloc.start()
    
    os.makedirs("output", exist_ok=True)
    matching_out_path = "output/matching_results.tsv"
    candidate_out_path = "output/candidate_pairs.tsv"
    
    # ------------------------------------------------------------------
    # PART 1: ULTRA-FAST STAGE 5 MODEL TRAINING (10,000 S1 ENTITIES)
    # ------------------------------------------------------------------
    print("\n--- PART 1: ULTRA-FAST STAGE 5 MODEL TRAINING ---")
    t0 = time.time()
    
    print("1. Loading ground truth subset (10,000 entities)...")
    gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=10000)
    s1_ids = set(gt_df["source1_entity_id"].dropna())
    
    gt_matches = {}
    needed_s2s3 = set()
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        matches = [m.strip() for m in matched_str.split(",") if m.strip()]
        gt_matches[s1_id] = set(matches)
        needed_s2s3.update(matches)
        
    print("2. Streaming train_source1.tsv...")
    s1_rows = []
    for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=50000):
        sub = chunk[chunk["entity_id"].isin(s1_ids)].copy()
        if not sub.empty:
            s1_rows.append(normalize_dataframe(sub))
        del chunk, sub
    s1_df = pd.concat(s1_rows, ignore_index=True)
    del s1_rows
    
    print("3. Indexing S2 & S3 into SQLite DiskInvertedIndex...")
    train_db_path = "train_fast_blocking.db"
    train_blocking = DiskInvertedIndex(db_path=train_db_path, max_block_size=1500)
    
    def stream_into_db(filepath, max_rows=100000):
        rows_read = 0
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
            train_blocking.build(records)
            rows_read += len(chunk)
            del chunk, norm_chunk, records
            if rows_read >= max_rows:
                break
                
    stream_into_db("dataset/train/train_source2.tsv", max_rows=100000)
    stream_into_db("dataset/train/train_source3.tsv", max_rows=100000)
    train_blocking.finalize_index()
    
    print("4. Candidate Generation & Feature Extraction (max_cands=30)...")
    sub_pairs = []
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        cands = train_blocking.generate_candidates(
            s1_row["norm_name"], s1_row["norm_address"], s1_row["country"], max_cands=30
        )
        pos_matches = gt_matches.get(s1_id, set())
        all_cands = list(set(cands).union(pos_matches))
        
        cand_records = train_blocking.get_records_by_ids(all_cands)
        for cand_id, cand_info in cand_records.items():
            is_match = 1.0 if cand_id in pos_matches else 0.0
            sub_pairs.append({
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
            
    p_df = pd.DataFrame(sub_pairs)
    feats_df = extract_features_batch(p_df)
    X = feats_df.values.astype(np.float32)
    y = p_df["label"].values.astype(np.int8)
    s1_ids_arr = p_df["s1_id"].values
    cand_ids_arr = p_df["cand_id"].values
    del p_df, feats_df, sub_pairs, s1_df
    
    print(f"Extraction Complete: X shape = {X.shape}, Positives = {int(y.sum())}")
    
    print("5. Entity-level Train/Validation Split (80/20)...")
    unique_s1 = np.array(list(set(s1_ids_arr)))
    np.random.seed(42)
    np.random.shuffle(unique_s1)
    
    split_idx = int(0.8 * len(unique_s1))
    train_s1_set = set(unique_s1[:split_idx])
    val_s1_set = set(unique_s1[split_idx:])
    
    train_mask = np.isin(s1_ids_arr, list(train_s1_set))
    val_mask = np.isin(s1_ids_arr, list(val_s1_set))
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    
    val_df = pd.DataFrame({
        "s1_id": s1_ids_arr[val_mask],
        "cand_id": cand_ids_arr[val_mask],
        "label": y_val
    })
    del X, y
    
    print("6. Training HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=100,
        learning_rate=0.1,
        max_depth=6,
        min_samples_leaf=20,
        random_state=42
    )
    clf.fit(X_train, y_train)
    del X_train, y_train
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_df["prob"] = val_probs
    del X_val, y_val
    
    print("\n7. Sweeping Decision Threshold tau in [0.85, 0.99] in steps of 0.01...")
    best_threshold = 0.85
    best_macro_f05 = 0.0
    val_s1_list = list(val_s1_set)
    
    for thresh in np.arange(0.85, 0.995, 0.01):
        thresh = round(float(thresh), 2)
        pred_matches = val_df[val_df["prob"] >= thresh]
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
        print(f"  Threshold tau = {thresh:.2f} -> Validation Macro F0.5 = {macro_f05:.4f}")
        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = thresh
            
    print(f"\n>>> OPTIMAL DECISION THRESHOLD tau: {best_threshold:.2f} (Macro F0.5 = {best_macro_f05:.4f})")
    print(f"Stage 5 Training Time: {(time.time() - t0)/60:.2f} minutes")
    train_blocking.cleanup()
    
    # ------------------------------------------------------------------
    # PART 2: STAGE 7 FULL TEST INFERENCE WITH SAFETY NET GUARANTEE
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print(f"--- PART 2: STAGE 7 FULL INFERENCE (max_cands=30, tau={best_threshold:.2f}) ---")
    print("======================================================================")
    t_inf_start = time.time()
    
    print("1. Reading test S1 entity ID master order for 100% completion guarantee...")
    test_s1_df = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", usecols=["entity_id"])
    s1_master_order = list(test_s1_df["entity_id"].astype(str))
    total_test_entities = len(s1_master_order)
    print(f"   Total test S1 entities: {total_test_entities:,}")
    
    # Initialize in-memory dictionaries for 100% fallback safety net
    matching_dict = {s1_id: "" for s1_id in s1_master_order}
    candidate_dict = {s1_id: "" for s1_id in s1_master_order}
    
    # Write initial fallback files immediately so output files ALWAYS exist and pass validator!
    write_valid_output(matching_dict, candidate_dict, s1_master_order, matching_out_path, candidate_out_path)
    print("   [SAFETY NET ACTIVE] Initial valid fallback matching_results.tsv written!")
    
    print("\n2. Building SQLite DiskInvertedIndex for Test S2 & S3 Records...")
    test_db_path = "test_fast_inference.db"
    test_blocking = DiskInvertedIndex(db_path=test_db_path, max_block_size=1500)
    
    def index_test_file_streamed(filepath):
        print(f"   Streaming {filepath} into SQLite DB...")
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
            test_blocking.build(records)
            del chunk, norm_chunk, records
            
    index_test_file_streamed("dataset/test/test_source2.tsv")
    index_test_file_streamed("dataset/test/test_source3.tsv")
    print("   Finalizing SQLite Index...")
    test_blocking.finalize_index()
    print("   SQLite Test Record Store & Inverted Index built successfully!")
    
    print("\n3. Stream-processing test_source1.tsv (chunksize=10,000, max_cands=30)...")
    test_s1_reader = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", chunksize=10000)
    
    total_processed = 0
    total_matched_pairs = 0
    last_print_time = time.time()
    
    for s1_chunk in test_s1_reader:
        norm_s1_chunk = normalize_dataframe(s1_chunk)
        
        for _, s1_row in norm_s1_chunk.iterrows():
            s1_id = str(s1_row["entity_id"])
            
            cands = test_blocking.generate_candidates(
                s1_row["norm_name"], s1_row["norm_address"], s1_row["country"], max_cands=30
            )
            
            if cands:
                candidate_dict[s1_id] = ",".join(cands)
                cand_records = test_blocking.get_records_by_ids(cands)
                
                eval_list = []
                for cand_id, cand_info in cand_records.items():
                    eval_list.append({
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
                        "cand_id_col": cand_id
                    })
                    
                eval_df = pd.DataFrame(eval_list)
                eval_df["cand_id"] = eval_df["cand_id_col"]
                
                feats_df = extract_features_batch(eval_df)
                probs = clf.predict_proba(feats_df.values)[:, 1]
                
                matched_cands = [
                    eval_list[i]["cand_id"]
                    for i in range(len(eval_list))
                    if probs[i] >= best_threshold
                ]
                
                if matched_cands:
                    matching_dict[s1_id] = ",".join(matched_cands)
                    total_matched_pairs += len(matched_cands)
                else:
                    matching_dict[s1_id] = ""
            else:
                candidate_dict[s1_id] = ""
                matching_dict[s1_id] = ""
                
            total_processed += 1
            
        del s1_chunk, norm_s1_chunk
        
        # Flush safety net output to disk after every chunk!
        write_valid_output(matching_dict, candidate_dict, s1_master_order, matching_out_path, candidate_out_path)
        
        now = time.time()
        elapsed_mins = (now - t_inf_start) / 60
        rate = total_processed / elapsed_mins if elapsed_mins > 0 else 0
        rem_entities = total_test_entities - total_processed
        eta_mins = rem_entities / rate if rate > 0 else 0
        pct = (total_processed / total_test_entities) * 100
        
        print(f"[INFERENCE PROGRESS] {total_processed:,} / {total_test_entities:,} entities ({pct:.1f}%) | Elapsed: {elapsed_mins:.2f} mins | Pace: {rate:,.0f} ent/min | ETA: {eta_mins:.1f} mins | Matches: {total_matched_pairs:,}")
        last_print_time = now
        
    test_blocking.cleanup()
    
    total_elapsed = (time.time() - overall_start) / 60
    print("\n" + "="*70)
    print("SUCCESS: PIPELINE COMPLETE & FULL INFERENCE FINISHED!")
    print(f"Total Pipeline Runtime: {total_elapsed:.2f} minutes")
    print(f"Output files saved to:")
    print(f"  - {matching_out_path}")
    print(f"  - {candidate_out_path}")
    print("="*70)
    
    # Run official validator
    print("\nRunning official submission validator...")
    res = subprocess.run(["python", "utils/validate_submission.py"], capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print(res.stderr)

if __name__ == "__main__":
    main()