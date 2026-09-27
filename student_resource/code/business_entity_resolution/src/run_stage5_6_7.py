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

def main():
    print("======================================================================")
    print("=== UNIFIED PIPELINE: STAGE 5 TRAINING & STAGE 7 FULL TEST INFERENCE ===")
    print("======================================================================")
    
    overall_start = time.time()
    tracemalloc.start()
    
    os.makedirs("output", exist_ok=True)
    
    # ------------------------------------------------------------------
    # PART 1: STAGE 5 CLASSIFIER TRAINING & THRESHOLD SWEEP (180,000 ENTITIES)
    # ------------------------------------------------------------------
    print("\n--- PART 1: STAGE 5 MODEL TRAINING (180,000 ENTITY SUBSET) ---")
    t0 = time.time()
    
    print("1. Loading ground truth subset (180,000 entities)...")
    gt_df = pd.read_csv("dataset/train/train_ground_truth.tsv", sep="\t", nrows=180000)
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
    
    print("2. Loading train S1 and relevant S2/S3 records...")
    s1_df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
    s1_df = s1_df[s1_df["entity_id"].isin(s1_ids)].copy()
    s1_df = normalize_dataframe(s1_df)
    
    # Load first 250k S2/S3 plus ground truth needed S2/S3
    s2_df = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=250000)
    s3_df = pd.read_csv("dataset/train/train_source3.tsv", sep="\t", nrows=250000)
    s2_df = normalize_dataframe(s2_df)
    s3_df = normalize_dataframe(s3_df)
    
    s23_df = pd.concat([s2_df, s3_df], ignore_index=True)
    
    print("3. Building SQLite DiskInvertedIndex for Training...")
    train_db_path = "train_pipeline_blocking.db"
    train_blocking = DiskInvertedIndex(db_path=train_db_path)
    
    records = [
        (
            row["entity_id"],
            row["business_name"],
            row["norm_name"],
            row["norm_address"],
            row["country"],
            row["is_non_latin"]
        )
        for _, row in s23_df.iterrows()
    ]
    train_blocking.build(records)
    del s2_df, s3_df, s23_df, records
    
    print("4. Generating candidate pairs for training subset...")
    pairs_list = []
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row["entity_id"]
        cands = train_blocking.generate_candidates(
            s1_row["norm_name"], s1_row["norm_address"], s1_row["country"]
        )
        pos_matches = gt_matches.get(s1_id, set())
        all_cands = list(set(cands).union(pos_matches))
        
        cand_records = train_blocking.get_records_by_ids(all_cands)
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
    print(f"Generated {len(pairs_df)} training candidate pairs.")
    
    print("5. Extracting dense pair features...")
    features_df = extract_features_batch(pairs_df)
    X = features_df.values
    y = pairs_df["label"].values
    
    print("6. Entity-level Train/Validation Split (80/20)...")
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
    
    print("7. Training HistGradientBoostingClassifier...")
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_pairs_df["prob"] = val_probs
    
    print("\n8. Sweeping Decision Threshold tau in [0.85, 0.99] in steps of 0.01...")
    best_threshold = 0.85
    best_macro_f05 = 0.0
    val_s1_list = list(val_s1)
    
    for thresh in np.arange(0.85, 0.995, 0.01):
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
        print(f"  Threshold tau = {thresh:.2f} -> Validation Macro F0.5 = {macro_f05:.4f}")
        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = thresh
            
    print(f"\n>>> BEST THRESHOLD tau: {best_threshold:.2f} (Macro F0.5 = {best_macro_f05:.4f})")
    print(f"Stage 5 Training Time: {(time.time() - t0)/60:.2f} minutes")
    train_blocking.cleanup()
    
    # ------------------------------------------------------------------
    # PART 2: STAGE 7 FULL TEST SET INFERENCE (~1.73M ENTITIES)
    # ------------------------------------------------------------------
    print("\n======================================================================")
    print(f"--- PART 2: STAGE 7 FULL INFERENCE ON TEST SET (tau = {best_threshold:.2f}) ---")
    print("======================================================================")
    t_inf_start = time.time()
    
    print("1. Building SQLite DiskInvertedIndex for Full Test S2 & S3 Records...")
    test_db_path = "test_inference_blocking.db"
    test_blocking = DiskInvertedIndex(db_path=test_db_path)
    
    def index_test_file(filepath):
        print(f"   Indexing {filepath} into SQLite DB in chunks...")
        for chunk in pd.read_csv(filepath, sep="\t", chunksize=100000):
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
            
    index_test_file("dataset/test/test_source2.tsv")
    index_test_file("dataset/test/test_source3.tsv")
    print("   SQLite Test Record Store & Inverted Index built successfully!")
    
    print("2. Stream-processing test_source1.tsv and writing output TSV files...")
    matching_out_path = "output/matching_results.tsv"
    candidate_out_path = "output/candidate_pairs.tsv"
    
    f_match = open(matching_out_path, "w", encoding="utf-8")
    f_cand = open(candidate_out_path, "w", encoding="utf-8")
    
    f_match.write("source1_entity_id\tmatched_entity_ids\n")
    f_cand.write("source1_entity_id\tcandidate_entity_ids\n")
    
    test_s1_reader = pd.read_csv("dataset/test/test_source1.tsv", sep="\t", chunksize=20000)
    
    total_processed = 0
    total_matched_pairs = 0
    last_progress_time = time.time()
    
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
        
        now = time.time()
        # Print progress update
        elapsed_mins = (now - t_inf_start) / 60
        print(f"[INFERENCE PROGRESS] Processed {total_processed} S1 entities | Elapsed: {elapsed_mins:.2f} mins | Total Matches: {total_matched_pairs}")
        last_progress_time = now
        
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