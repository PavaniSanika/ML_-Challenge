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
    print('=== STAGE 5: Classifier Training and F0.5 Threshold Tuning (50,000 Sample) ===')
    tracemalloc.start()
    start_time = time.time()
    
    print('1. Loading sample ground truth (50,000 entities)...')
    gt_df = pd.read_csv('dataset/train/train_ground_truth.tsv', sep='\t', nrows=50000)
    s1_ids = set(gt_df['source1_entity_id'].dropna())
    
    gt_matches = {}
    needed_s2s3 = set()
    for _, row in gt_df.iterrows():
        s1_id = row['source1_entity_id']
        matched_str = str(row['matched_entity_ids']) if pd.notna(row['matched_entity_ids']) else ''
        matches = [m.strip() for m in matched_str.split(',') if m.strip()]
        gt_matches[s1_id] = set(matches)
        needed_s2s3.update(matches)
        
    print(f'Loaded {len(s1_ids)} S1 entities and {len(gt_matches)} ground truth match sets.')
    
    print('2. Loading source data files...')
    s1_df = pd.read_csv('dataset/train/train_source1.tsv', sep='\t')
    s1_df = s1_df[s1_df['entity_id'].isin(s1_ids)].copy()
    
    s2_df = pd.read_csv('dataset/train/train_source2.tsv', sep='\t')
    s3_df = pd.read_csv('dataset/train/train_source3.tsv', sep='\t')
    
    s2_df = s2_df[s2_df['entity_id'].isin(needed_s2s3) | (s2_df.index < 200000)].copy()
    s3_df = s3_df[s3_df['entity_id'].isin(needed_s2s3) | (s3_df.index < 200000)].copy()
    
    print(f'Sample S1 records: {len(s1_df)}, Sample S2 records: {len(s2_df)}, Sample S3 records: {len(s3_df)}')
    
    print('3. Normalizing business records...')
    s1_df = normalize_dataframe(s1_df)
    s2_df = normalize_dataframe(s2_df)
    s3_df = normalize_dataframe(s3_df)
    
    s2s3_df = pd.concat([s2_df, s3_df], ignore_index=True)
    s2s3_map = s2s3_df.set_index('entity_id').to_dict(orient='index')
    
    print('4. Building SQLite DiskInvertedIndex...')
    db_path = 'train_sample_blocking.db'
    blocking_index = DiskInvertedIndex(db_path=db_path)
    
    s2s3_records = [
        (row['entity_id'], row['norm_name'], row['norm_address'], row['country'])
        for _, row in s2s3_df.iterrows()
    ]
    blocking_index.build(s2s3_records)
    print('DiskInvertedIndex built successfully!')
    
    print('5. Generating candidates and extracting pair features...')
    pairs_list = []
    
    for _, s1_row in s1_df.iterrows():
        s1_id = s1_row['entity_id']
        cands = blocking_index.generate_candidates(
            s1_row['norm_name'], s1_row['norm_address'], s1_row['country']
        )
        
        pos_matches = gt_matches.get(s1_id, set())
        all_cands = set(cands).union(pos_matches)
        
        for cand_id in all_cands:
            if cand_id in s2s3_map:
                cand_info = s2s3_map[cand_id]
                is_match = 1.0 if cand_id in pos_matches else 0.0
                pairs_list.append({
                    's1_id': s1_id,
                    'cand_id': cand_id,
                    'name1': s1_row['norm_name'],
                    'raw1': s1_row['business_name'],
                    'addr1': s1_row['norm_address'],
                    'country1': s1_row['country'],
                    'nl1': s1_row['is_non_latin'],
                    'name2': cand_info['norm_name'],
                    'raw2': cand_info['business_name'],
                    'addr2': cand_info['norm_address'],
                    'country2': cand_info['country'],
                    'nl2': cand_info['is_non_latin'],
                    'label': is_match
                })
                
    pairs_df = pd.DataFrame(pairs_list)
    print(f'Generated {len(pairs_df)} candidate pairs.')
    
    features_df = extract_features_batch(pairs_df)
    X = features_df.values
    y = pairs_df['label'].values
    
    print('6. Entity-level Train/Validation Split (80/20)...')
    unique_s1 = np.array(list(s1_ids))
    np.random.seed(42)
    np.random.shuffle(unique_s1)
    
    split_idx = int(0.8 * len(unique_s1))
    train_s1 = set(unique_s1[:split_idx])
    val_s1 = set(unique_s1[split_idx:])
    
    train_mask = pairs_df['s1_id'].isin(train_s1).values
    val_mask = pairs_df['s1_id'].isin(val_s1).values
    
    X_train, y_train = X[train_mask], y[train_mask]
    X_val, y_val = X[val_mask], y[val_mask]
    val_pairs_df = pairs_df[val_mask].copy()
    
    print(f'Train pairs: {len(X_train)} ({int(y_train.sum())} pos), Val pairs: {len(X_val)} ({int(y_val.sum())} pos)')
    
    print('7. Training HistGradientBoostingClassifier...')
    clf = HistGradientBoostingClassifier(
        max_iter=150,
        learning_rate=0.08,
        max_depth=8,
        min_samples_leaf=20,
        random_state=42
    )
    clf.fit(X_train, y_train)
    
    val_probs = clf.predict_proba(X_val)[:, 1]
    val_pairs_df['prob'] = val_probs
    
    print('8. Tuning Decision Threshold for Macro F0.5 Score...')
    best_threshold = 0.5
    best_macro_f05 = 0.0
    
    val_s1_list = list(val_s1)
    
    for thresh in np.arange(0.1, 0.95, 0.05):
        thresh = round(float(thresh), 2)
        
        pred_matches = val_pairs_df[val_pairs_df['prob'] >= thresh]
        pred_dict = pred_matches.groupby('s1_id')['cand_id'].apply(set).to_dict()
        
        f05_scores = []
        for s1_id in val_s1_list:
            gt_set = gt_matches.get(s1_id, set())
            pred_set = pred_dict.get(s1_id, set())
            
            if not gt_set and not pred_set:
                f05_scores.append(1.0)
            elif not pred_set and gt_set:
                f05_scores.append(0.0)
            elif pred_set and not gt_set:
                f05_scores.append(0.0)
            else:
                tp = len(gt_set.intersection(pred_set))
                fp = len(pred_set - gt_set)
                fn = len(gt_set - pred_set)
                
                prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f05 = compute_f05(prec, rec)
                f05_scores.append(f05)
                
        macro_f05 = float(np.mean(f05_scores))
        print(f'Threshold = {thresh:.2f} -> Validation Macro F0.5 = {macro_f05:.4f}')
        
        if macro_f05 > best_macro_f05:
            best_macro_f05 = macro_f05
            best_threshold = thresh
            
    print(f'\nOPTIMAL DECISION THRESHOLD: {best_threshold:.2f} (Best Macro F0.5 = {best_macro_f05:.4f})')
    
    blocking_index.cleanup()
    
    elapsed = time.time() - start_time
    current_mem, peak_mem = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    
    print('\n' + '='*60)
    print(f'STAGE 5 SANITY RUN COMPLETE!')
    print(f'Total Execution Time: {elapsed:.2f} seconds ({elapsed/60:.2f} mins)')
    print(f'Peak Memory Usage: {peak_mem / (1024 * 1024):.2f} MB')
    print('='*60)

if __name__ == '__main__':
    main()
