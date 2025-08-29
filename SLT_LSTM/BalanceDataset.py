#BalanceDataset

version = 'church'

import pandas as pd
def redistributeDataset(version):
    # Load current datasets
    test = pd.read_csv(f"testss_{version}.csv")
    train = pd.read_csv(f"trainss_{version}.csv")
    val = pd.read_csv(f"valss_{version}.csv")
    
    total = len(train) + len(test) + len(val)

    print("Original distribution:")
    print(f"Training: {len(train)} samples ({len(train)/total*100:.1f}%)")
    print(f"Testing: {len(test)} samples ({len(test)/total*100:.1f}%)")
    print(f"Validation: {len(val)} samples ({len(val)/total*100:.1f}%)")
    print(f"Total: {len(train) + len(test) + len(val)} samples")
    
    # Analyze Gloss distribution in training set
    train_label_counts = train['Gloss'].value_counts()
    test_label_counts = test['Gloss'].value_counts()
    
    # Calculate the target split with validation at exactly 12%
    total_samples = len(train) + len(test) + len(val)
    validation_percent = len(val) / total_samples
    print(f"Current validation percentage: {validation_percent*100:.1f}%")
    
    # Keep validation at current size, distribute remaining between train/test
    remaining_samples = total_samples - len(val)
    target_train_size = int(remaining_samples * 0.8)  # 80% of non-validation data
    samples_to_move = target_train_size - len(train)
    
    # Ensure we maintain a reasonable testing set size (at least 15% of total)
    min_test_size = int(total_samples * 0.15)
    max_samples_to_move = len(test) - min_test_size
    samples_to_move = min(samples_to_move, max_samples_to_move)
    
    print(f"\nPlanning to copy {samples_to_move} samples from test to train")
    print(f"This will result in approximately: {(len(train) + samples_to_move) / total_samples * 100:.1f}% train, " 
          f"{(len(test)- samples_to_move) / total_samples * 100:.1f}% test, {len(val) / total_samples * 100:.1f}% validation")
    
    # Identify underrepresented labels in training set
    # Calculate the ratio of training to testing samples for each label
    label_ratios = {}
    for label in set(train['Gloss'].unique()).union(set(test['Gloss'].unique())):
        train_count = train_label_counts.get(label, 0)
        test_count = test_label_counts.get(label, 0)
        if train_count + test_count > 0:
            label_ratios[label] = train_count / (train_count + test_count)
    
    # Find labels that are most underrepresented in training
    sorted_labels = sorted(label_ratios.items(), key=lambda x: x[1])
    underrepresented_labels = [label for label, ratio in sorted_labels if ratio < 0.5]
    
    print(f"\nUnderrepresented labels in training set: {underrepresented_labels[:10]}...")
    
    # Create copies of the dataframes to modify
    train_new = train.copy()
    
    # Count samples moved
    moved_samples = 0
    
    # First, try to balance underrepresented labels
    print("\nCopying samples for underrepresented labels:")
    for label in underrepresented_labels:
        if moved_samples >= samples_to_move:
            break
            
        label_test_samples = test[test['Gloss'] == label]
        
        # Calculate how many to move for this label
        # Try to make training have at least 70% of this label's samples
        total_label_samples = len(train[train['Gloss'] == label]) + len(label_test_samples)
        target_label_train = int(total_label_samples * 0.7)
        current_label_train = len(train[train['Gloss'] == label])
        label_samples_to_move = min(
            target_label_train - current_label_train,  # How many needed to reach 70%
            len(label_test_samples),                   # How many available
            samples_to_move - moved_samples            # How many we still need to move total
        )
        
        if label_samples_to_move > 0:
            # Copy samples from test to train
            samples_to_transfer = label_test_samples.sample(n=label_samples_to_move)
            
            # Add to training set
            train_new = pd.concat([train_new, samples_to_transfer])
            
            # Update counter
            moved_samples += label_samples_to_move
            
            print(f"  {label}: copied {label_samples_to_move} samples")
    
    # If we haven't moved enough samples yet, get more from other classes
    if moved_samples < samples_to_move:
        remaining_to_move = samples_to_move - moved_samples
        print(f"\nCopying {remaining_to_move} additional samples from test to balance classes:")
        
        # Get counts of remaining test samples by class
        remaining_test_counts = test['Gloss'].value_counts()
        
        # Move proportionally from classes
        for label, count in remaining_test_counts.items():
            # Skip already processed underrepresented labels
            if label in underrepresented_labels:
                continue
                
            # Calculate proportional amount to move
            proportion = count / sum(remaining_test_counts[remaining_test_counts.index.difference(underrepresented_labels)])
            label_to_move = min(int(remaining_to_move * proportion) + 1, count)
            label_to_move = min(label_to_move, remaining_to_move)
            
            if label_to_move > 0:
                # Copy samples from test to train
                label_test_samples = test[test['Gloss'] == label]
                samples_to_transfer = label_test_samples.sample(n=label_to_move)
                
                # Add to training set
                train_new = pd.concat([train_new, samples_to_transfer])
                
                # Update counter
                moved_samples += label_to_move
                remaining_to_move -= label_to_move
                
                print(f"  {label}: copied {label_to_move} samples")
                
            if remaining_to_move <= 0:
                break
    
    # Save the updated training dataset
    print("\nSaving updated training dataset...")
    train_new.to_csv(f"trainss_{version}.csv", index=False)
    
    # Print final distribution
    print("\nFinal distribution:")
    print(f"Training: {len(train_new)} samples ({len(train_new)/total_samples*100:.1f}%)")
    print(f"Testing: {len(test)} samples ({len(test)/total_samples*100:.1f}%)")
    print(f"Validation: {len(val)} samples ({len(val)/total_samples*100:.1f}%)")
    print(f"Total: {len(train_new) + len(test) + len(val)} samples")
    
    # Print improvement in class representation
    print("\nLabel representation improvement in training:")
    for label in underrepresented_labels[:10]:  # Show top 10 most underrepresented
        before = train_label_counts.get(label, 0)
        after = train_new['Gloss'].value_counts().get(label, 0)
        change = after - before
        if before > 0:
            percent_increase = (change / before) * 100
            print(f"{label}: {before} → {after} (+{change}, +{percent_increase:.1f}%)")
        else:
            print(f"{label}: {before} → {after} (+{change}, new class added)")
            
    # Check if any classes are now better balanced
    print("\nFinal class balance (train/total ratio):")
    for label in sorted_labels[:10]:  # Show top 10 initially most underrepresented
        label_name = label[0]
        initial_ratio = label[1]
        train_count = train_new['Gloss'].value_counts().get(label_name, 0)
        test_count = test['Gloss'].value_counts().get(label_name, 0)
        total_count = train_count + test_count
        if total_count > 0:
            new_ratio = train_count / total_count
            print(f"{label_name}: {initial_ratio:.2f} → {new_ratio:.2f}")


#get values present in training and testing and delete the entries from test
def removeDuplicates(version):
    train = pd.read_csv(f"trainss_{version}.csv")
    test = pd.read_csv(f"testss_{version}.csv")
    val = pd.read_csv(f"valss_{version}.csv")
    
    print("Original distribution:")
    print(f"Training: {len(train)} samples")
    print(f"Testing: {len(test)} samples")
    print(f"Validation: {len(val)} samples")
    print(f"Total: {len(train) + len(test) + len(val)} samples")
    
    # Check for exact duplicate video files across datasets
    train_files = set(train['Video file'])
    test_files = set(test['Video file'])
    val_files = set(val['Video file'])
    
    # Find duplicates
    train_test_duplicates = train_files.intersection(test_files)
    train_val_duplicates = train_files.intersection(val_files)
    test_val_duplicates = test_files.intersection(val_files)
    
    print(f"\nFound {len(train_test_duplicates)} duplicates between train and test")
    print(f"Found {len(train_val_duplicates)} duplicates between train and val")
    print(f"Found {len(test_val_duplicates)} duplicates between test and val")
    
    # Remove duplicates (prioritize keeping them in training set)
    if len(train_test_duplicates) > 0:
        test = test[~test['Video file'].isin(train_test_duplicates)]
        
    if len(train_val_duplicates) > 0:
        val = val[~val['Video file'].isin(train_val_duplicates)]
        
    if len(test_val_duplicates) > 0:
        val = val[~val['Video file'].isin(test_val_duplicates)]
    
    # Save updated datasets
    train.to_csv(f"trainss_{version}.csv", index=False)
    test.to_csv(f"testss_{version}.csv", index=False)
    val.to_csv(f"valss_{version}.csv", index=False)
    
    print("\nAfter removing duplicates:")
    print(f"Training: {len(train)} samples")
    print(f"Testing: {len(test)} samples")
    print(f"Validation: {len(val)} samples")
    print(f"Total: {len(train) + len(test) + len(val)} samples")
    
    # Check class distribution
    all_classes = set(train['Gloss'].unique()).union(
                  set(test['Gloss'].unique())).union(
                  set(val['Gloss'].unique()))
    
    print(f"\nTotal unique classes: {len(all_classes)}")
    print(f"Classes in training: {len(train['Gloss'].unique())}")
    print(f"Classes in testing: {len(test['Gloss'].unique())}")
    print(f"Classes in validation: {len(val['Gloss'].unique())}")
    
    # Find classes missing from test but present in train
    missing_from_test = set(train['Gloss'].unique()) - set(test['Gloss'].unique())
    if missing_from_test:
        print(f"\nWarning: {len(missing_from_test)} classes in training but missing from test")
        print(f"Examples: {list(missing_from_test)[:5]}")

removeDuplicates(version)

# Run the redistribution
redistributeDataset(version)

removeDuplicates(version)
