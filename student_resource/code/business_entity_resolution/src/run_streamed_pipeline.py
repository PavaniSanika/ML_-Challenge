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
    print("======================================================================")
    print("=== ZERO-RAM STREAMED PIPELINE: STAGE 5 TRAINING & STAGE 7 INFERENCE ===")
    print("======================================================================")
    
    overall_start = time.time()
    tracemalloc.start()
    
    os.makedirs("output", exist_ok=True)
    
    # ------------------------------------------------------------------
    # PART 1: STAGE 5 STREAMED TRAINING (50,000 S1 ENTITIES)
    # ------------------------------------------------------------------
    print("\n--- PART 1: STAGE 5 STREAMED MODEL TRAINING (50,000 S1 ENTITIES) ---")
    t0 = time.time()
    
    print("1. Loading ground truth subset (50,000 entities)...")
    gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=50000)
    s1_ids = set(gt_df["source1_entity_id"].dropna())
    
    gt_matches = {}
    needed_s2s3 = set()
    for _, row in gt_df.iterrows():
        s1_id = str(row["source1_entity_id"])
        matched_str = str(row["matched_entity_ids"]) if pd.notna(row["matched_entity_ids"]) else ""
        matches = [m.strip() for m in matched_str.split(",") if m.strip()]
        gt_matches[s1_id] = set(matches)
        needed_s2s3.update(matches)
        
    print(f"Loaded {len(s1_ids)} S1 entities and {len(gt_matches)} ground truth match sets.")
    cur, peak = get_mem_mb()
    print(f"[MEM CHECKPOINT] Ground truth loaded: {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    print("\n2. Streaming train_source1.tsv in 50,000-row chunks...")
    s1_rows = []
    for chunk in pd.read_csv("dataset/train/train_source1.tsv", sep="\t", chunksize=50000):
        sub = chunk[chunk["entity_id"].isin(s1_ids)].copy()
        if not sub.empty:
            norm_sub = normalize_dataframe(sub)
            s1_rows.append(norm_sub)
        del chunk, sub
    s1_df = pd.concat(s1_rows, ignore_index=True)
    del s1_rows
    cur, peak = get_mem_mb()
    print(f"[MEM CHECKPOINT] S1 subset loaded & normalized ({len(s1_df)} rows): {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    print("\n3. Streaming train S2 & S3 into SQLite DiskInvertedIndex...")
    train_db_path = "train_pipeline_blocking.db"
    train_blocking = DiskInvertedIndex(db_path=train_db_path)
    
    def stream_into_db(filepath, max_rows=150000):
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
                
    stream_into_db("dataset/train/train_source2.tsv", max_rows=150000)
    stream_into_db("dataset/train/train_source3.tsv", max_rows=150000)
    
    cur, peak = get_mem_mb()
    print(f"[MEM CHECKPOINT] SQLite Train Index Built: {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    print("\n4. Chunked Candidate Generation & Feature Extraction...")
    chunk_size = 5000
    X_chunks = []
    y_chunks = []
    s1_id_chunks = []
    cand_id_chunks = []
    
    for i in range(0, len(s1_df), chunk_size):
        sub_s1 = s1_df.iloc[i:i+chunk_size]
        sub_pairs = []
        
        for _, s1_row in sub_s1.iterrows():
            s1_id = s1_row["entity_id"]
            cands = train_blocking.generate_candidates(
                s1_row["norm_name"], s1_row["norm_address"], s1_row["country"]
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
                
        if sub_pairs:
            p_df = pd.DataFrame(sub_pairs)
            feats_df = extract_features_batch(p_df)
            X_chunks.append(feats_df.values.astype(np.float32))
            y_chunks.append(p_df["label"].values.astype(np.int8))
            s1_id_chunks.extend(p_df["s1_id"].values)
            cand_id_chunks.extend(p_df["cand_id"].values)
            del p_df, feats_df, sub_pairs
            
    X = np.vstack(X_chunks)
    y = np.concatenate(y_chunks)
    s1_ids_arr = np.array(s1_id_chunks)
    cand_ids_arr = np.array(cand_id_chunks)
    
    del X_chunks, y_chunks, s1_id_chunks, cand_id_chunks, s1_df
    cur, peak = get_mem_mb()
    print(f"[MEM CHECKPOINT] Feature Extraction Complete (X shape = {X.shape}): {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    print("\n5. Entity-level Train/Validation Split (80/20)...")
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
    del X, y, s1_ids_arr, cand_ids_arr
    
    print("\n6. Training HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.08,
        max_depth=8,
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
    # PART 2: STAGE 7 FULL TEST SET INFERENCE (~1.73M ENTITIES)
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print(f"--- PART 2: STAGE 7 FULL INFERENCE ON TEST SET (tau = {best_threshold:.2f}) ---")
    print("======================================================================")
    t_inf_start = time.time()
    
    print("1. Building SQLite DiskInvertedIndex for Full Test S2 & S3 Records in 50,000-row chunks...")
    test_db_path = "test_inference_blocking.db"
    test_blocking = DiskInvertedIndex(db_path=test_db_path)
    
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
    print("   SQLite Test Record Store & Inverted Index built successfully!")
    cur, peak = get_mem_mb()
    print(f"[MEM CHECKPOINT] SQLite Test DB Index Built: {cur:.2f} MB (Peak: {peak:.2f} MB)")
    
    print("2. Stream-processing test_source1.tsv and writing output TSV files...")
    matching_out_path = "output/matching_results.tsv"
    candidate_out_path = "output/candidate_pairs.tsv"
    
    f_match = open(matching_out_path, "w", encoding="utf-8")
    f_cand = open(candidate_out_path, "w", encoding="utf-8")
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    test_s1_reader = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", chunksize=10000)
    
    total_processed = 0
    total_matched_pairs = 0
    last_print_time = time.time()
    
    for s1_chunk in test_s1_reader:
        norm_s1_chunk = normalize_dataframe(s1_chunk)
        
        for _, s1_row in norm_s1_chunk.iterrows():
            s1_id = s1_row["entity_id"]
            
            cands = test_blocking.generate_candidates(
                s1_row["norm_name"], s1_row["norm_address"], s1_row["country"]
            )
            
            if not cands:
                f_cand.write(f"{s1_id}\t\n")
                f_match.write(f"{s1_id}\t\n")
            else:
                f_cand.write(f"{s1_id}\t{','.join(cands)}\n")
                
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
                    f_match.write(f"{s1_id}\t{','.join(matched_cands)}\n")
                    total_matched_pairs += len(matched_cands)
                else:
                    f_match.write(f"{s1_id}\t\n")
                    
            total_processed += 1
            
        f_match.flush()
        f_cand.flush()
        del s1_chunk, norm_s1_chunk
        
        now = time.time()
        if now - last_print_time >= 120 or total_processed % 50000 == 0:
            elapsed_mins = (now - t_inf_start) / 60
            pct = (total_processed / 1730000) * 100
            print(f"[INFERENCE PROGRESS] {total_processed} / 1730000 entities ({pct:.1f}%) | Elapsed: {elapsed_mins:.2f} mins | Total Matches: {total_matched_pairs}")
            last_print_time = now
            
    f_match.close()
    f_cand.close()
    test_blocking.cleanup()
    
    total_elapsed = (time.time() - overall_start) / 60
    print("\n" + "="*70)
    print("SUCCESS: PIPELINE COMPLETE & INFERENCE FINISHED!")
    print(f"Total Pipeline Runtime: {total_elapsed:.2f} minutes")
    print(f"Output files saved to:")
    print(f"  - {matching_out_path}")
    print(f"  - {candidate_out_path}")
    print("="*70)

if __name__ == "__main__":
    main()