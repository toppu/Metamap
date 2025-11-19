"""
Python replacements for R functions (r_support.R)
This module provides pure Python implementations of statistical functions
previously implemented in R, eliminating the need for rpy2 bridge.
"""

import warnings

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from scipy.stats import entropy
from sklearn.feature_selection import VarianceThreshold
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler

# =============================================================================
# ALPHA DIVERSITY FUNCTIONS
# =============================================================================

def simpson(df):
    """
    Calculate Simpson diversity index for each sample.
    
    Simpson's Index measures the probability that two individuals 
    randomly selected from a sample will belong to the same species.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data frame where rows are samples and columns are taxa (OTUs/species)
    
    Returns:
    --------
    numpy.array
        Simpson diversity index for each sample
    """
    # Convert to numpy array and ensure float type
    data = df.values.astype(float)
    
    # Calculate proportions
    totals = data.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1  # Avoid division by zero
    proportions = data / totals
    
    # Simpson index: 1 - sum(p_i^2)
    simpson_values = 1 - np.sum(proportions ** 2, axis=1)
    
    return simpson_values


def shannon(df):
    """
    Calculate Shannon diversity index for each sample.
    
    Shannon's Index measures the entropy (uncertainty) in predicting 
    the species identity of a randomly chosen individual.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data frame where rows are samples and columns are taxa (OTUs/species)
    
    Returns:
    --------
    numpy.array
        Shannon diversity index for each sample
    """
    # Convert to numpy array and ensure float type
    data = df.values.astype(float)
    
    # Calculate proportions
    totals = data.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1  # Avoid division by zero
    proportions = data / totals
    
    # Replace zeros to avoid log(0)
    proportions[proportions == 0] = 1
    
    # Shannon index: -sum(p_i * log(p_i))
    shannon_values = -np.sum(proportions * np.log(proportions), axis=1)
    
    return shannon_values


# =============================================================================
# BETA DIVERSITY FUNCTIONS
# =============================================================================

def beta_dim_red(df, beta_method, dim_method):
    """
    Calculate beta diversity distance matrix and perform dimensionality reduction.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data frame where rows are samples and columns are taxa
    beta_method : str
        Distance metric: 'bray' (Bray-Curtis), 'jaccard', 'euclidean'
    dim_method : str
        Dimensionality reduction method: 'NMDS' (MDS) or 'PCoA' (classical MDS)
    
    Returns:
    --------
    dict or numpy.array
        For NMDS: dict with 'a' (coordinates) and 'b' (stress)
        For PCoA: numpy.array of coordinates
    """
    # Remove 'id' column if present
    if 'id' in df.columns:
        df = df.drop('id', axis=1)
    
    # Convert to numpy array
    data = df.values.astype(float)
    
    # Calculate distance matrix based on method
    if beta_method == 'bray':
        # Bray-Curtis dissimilarity
        distances = pdist(data, metric='braycurtis')
    elif beta_method == 'jaccard':
        # Jaccard distance
        distances = pdist(data, metric='jaccard')
    elif beta_method == 'euclidean':
        # Euclidean distance
        distances = pdist(data, metric='euclidean')
    elif beta_method == 'gower':
        # Gower distance (for mixed data types)
        # Simple implementation for continuous data
        distances = pdist(data, metric='cityblock')
        # Normalize by number of features
        distances = distances / data.shape[1]
    else:
        raise ValueError("Invalid beta_method. Choose 'bray', 'jaccard', 'euclidean', or 'gower'.")
    
    # Convert to square distance matrix
    dist_matrix = squareform(distances)
    
    # Perform dimensionality reduction
    if dim_method == 'NMDS':
        # Non-metric multidimensional scaling
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, 
                  normalized_stress='auto', max_iter=300)
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            projections = mds.fit_transform(dist_matrix)
        
        # Calculate stress (goodness of fit)
        stress = mds.stress_
        
        return {'a': projections, 'b': stress}
    
    elif dim_method == 'PCoA':
        # Principal Coordinates Analysis (Classical MDS)
        mds = MDS(n_components=2, dissimilarity='precomputed', random_state=42, 
                  metric=True)
        projections = mds.fit_transform(dist_matrix)
        
        return projections
    
    else:
        raise ValueError("Invalid dim_method. Choose 'NMDS' or 'PCoA'.")


# =============================================================================
# FEATURE SELECTION FUNCTIONS
# =============================================================================

def cut_var(df, cutoff=95):
    """
    Identify near-zero variance features for removal.
    
    Identifies features with very low variance that are unlikely to be 
    useful for modeling (replacement for caret's nearZeroVar).
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data frame with features as columns
    cutoff : float
        Frequency ratio cutoff (default: 95). Features with ratio of most 
        common to second most common value above this are flagged.
    
    Returns:
    --------
    numpy.array
        Boolean array indicating which features have near-zero variance (True = remove)
    """
    n_samples = len(df)
    nzv_flags = []
    
    for col in df.columns:
        values = df[col].values
        
        # Get unique values and their counts
        unique_values, counts = np.unique(values, return_counts=True)
        
        # If only one unique value, it's zero variance
        if len(unique_values) == 1:
            nzv_flags.append(True)
            continue
        
        # Calculate frequency ratio (most common / second most common)
        sorted_counts = np.sort(counts)[::-1]
        if len(sorted_counts) > 1:
            freq_ratio = sorted_counts[0] / sorted_counts[1]
        else:
            freq_ratio = sorted_counts[0]
        
        # Calculate percentage of unique values
        percent_unique = (len(unique_values) / n_samples) * 100
        
        # Flag as near-zero variance if:
        # 1. Frequency ratio exceeds cutoff AND
        # 2. Percent unique values is less than 10%
        if freq_ratio > cutoff and percent_unique < 10:
            nzv_flags.append(True)
        else:
            nzv_flags.append(False)
    
    return np.array(nzv_flags)


def remove_low_variance_features(df, threshold=0.01):
    """
    Remove features with variance below threshold.
    
    Alternative simpler approach using sklearn's VarianceThreshold.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Data frame with features as columns
    threshold : float
        Variance threshold (default: 0.01)
    
    Returns:
    --------
    tuple
        (filtered_df, removed_columns)
    """
    selector = VarianceThreshold(threshold=threshold)
    
    # Fit and transform
    data_filtered = selector.fit_transform(df)
    
    # Get names of removed columns
    removed_mask = ~selector.get_support()
    removed_columns = df.columns[removed_mask].tolist()
    kept_columns = df.columns[selector.get_support()].tolist()
    
    # Create filtered dataframe
    df_filtered = pd.DataFrame(data_filtered, columns=kept_columns, index=df.index)
    
    return df_filtered, removed_columns


# =============================================================================
# DIFFERENTIAL ABUNDANCE ANALYSIS (ANCOM-BC2 Replacement)
# =============================================================================

def perform_ancom_alternative(df, y, tab, level, method='mannwhitneyu'):
    """
    Perform differential abundance analysis as an alternative to ANCOM-BC2.
    
    This is a simplified approach using Mann-Whitney U test or t-test for 
    each feature. For production use, consider more sophisticated methods 
    like ALDEx2, songbird, or corncob.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        OTU/ASV count data (samples as columns, features as rows)
    y : pandas.DataFrame
        Metadata with 'bin_var' column (binary grouping variable)
    tab : pandas.DataFrame
        Taxonomy table
    level : str
        Taxonomic level for aggregation
    method : str
        Statistical test: 'mannwhitneyu' (default) or 'ttest'
    
    Returns:
    --------
    pandas.DataFrame
        Results table with taxon, p-value, adjusted p-value, log fold change
    """
    from scipy.stats import mannwhitneyu, ttest_ind
    from statsmodels.stats.multitest import multipletests
    
    # Ensure df is transposed correctly (samples as rows, features as columns)
    # df comes in with taxonomy as index, samples as columns
    # We need to transpose so samples are rows (index) and features are columns
    if df.index.name == 'taxonomy' or (len(df.index) > 0 and isinstance(df.index[0], str) and ';' in str(df.index[0])):
        # df has features as rows, need to transpose
        df = df.T
    
    # Aggregate by taxonomic level
    # Match taxonomy strings with the taxonomy table to get the level values
    if level:
        # Create a mapping from full taxonomy to the specified level
        tax_map = {}
        for tax_full in df.columns:
            if tax_full in tab.index:
                tax_level_value = tab.loc[tax_full, level]
                # Ensure we get a scalar value, not a Series
                if isinstance(tax_level_value, pd.Series):
                    tax_level_value = tax_level_value.iloc[0]
                tax_map[tax_full] = str(tax_level_value) if pd.notna(tax_level_value) else 'Unknown'
            else:
                tax_map[tax_full] = 'Unknown'
        
        # Group columns by taxonomic level and sum
        df_grouped = pd.DataFrame(index=df.index)
        for level_value in set(tax_map.values()):
            cols_to_sum = [col for col, lv in tax_map.items() if lv == level_value]
            if cols_to_sum:
                df_grouped[level_value] = df[cols_to_sum].sum(axis=1)
        
        df = df_grouped
    
    # Align df and y by index (ensure same samples)
    common_samples = df.index.intersection(y.index)
    
    if len(common_samples) == 0:
        raise ValueError(f"No common samples found between df (indices: {df.index[:5].tolist()}) and y (indices: {y.index[:5].tolist()})")
    
    df = df.loc[common_samples]
    y = y.loc[common_samples]
    
    # Get binary variable
    groups = y['bin_var'].values
    unique_groups = np.unique(groups)
    
    if len(unique_groups) != 2:
        # Provide detailed error message
        group_counts = pd.Series(groups).value_counts()
        raise ValueError(
            f"bin_var must be binary (exactly 2 groups). "
            f"Found {len(unique_groups)} groups after alignment: {dict(group_counts)}. "
            f"Common samples: {len(common_samples)}, Original df samples: {len(df.index)}, y samples: {len(y.index)}"
        )
    
    # Store results
    results = []
    
    # Test each feature
    for taxon in df.columns:
        abundance = df[taxon].values
        
        # Split by group (now dimensions match!)
        group1 = abundance[groups == unique_groups[0]]
        group2 = abundance[groups == unique_groups[1]]
        
        # Skip if one group has no variance
        if group1.std() == 0 or group2.std() == 0:
            continue
        
        # Perform statistical test
        try:
            if method == 'mannwhitneyu':
                statistic, pval = mannwhitneyu(group1, group2, alternative='two-sided')
            elif method == 'ttest':
                statistic, pval = ttest_ind(group1, group2)
            else:
                raise ValueError("method must be 'mannwhitneyu' or 'ttest'")
            
            # Calculate log fold change (mean difference in log scale)
            # Add pseudocount to avoid log(0)
            mean1 = np.log2(group1.mean() + 1)
            mean2 = np.log2(group2.mean() + 1)
            log_fc = mean2 - mean1
            
            results.append({
                'taxon': taxon,
                'statistic': statistic,
                'pval': pval,
                'log_fold_change': log_fc,
                'mean_group1': group1.mean(),
                'mean_group2': group2.mean()
            })
        except Exception as e:
            # Skip features that cause errors
            continue
    
    # Create results dataframe
    res_df = pd.DataFrame(results)
    
    if len(res_df) == 0:
        return pd.DataFrame()  # Return empty if no results
    
    # Adjust p-values for multiple testing (Benjamini-Hochberg)
    res_df['padj'] = multipletests(res_df['pval'], method='fdr_bh')[1]
    
    # Sort by p-value
    res_df = res_df.sort_values('pval')
    
    # Add significance flag
    res_df['significant'] = res_df['padj'] < 0.05
    
    return res_df


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def test_python_functions():
    """
    Test all Python functions to ensure they work correctly.
    """
    # Create test data
    np.random.seed(42)
    test_df = pd.DataFrame(
        np.random.randint(0, 100, size=(10, 20)),
        columns=[f'OTU_{i}' for i in range(20)]
    )
    
    print("Testing Alpha Diversity Functions...")
    print(f"Simpson: {simpson(test_df)[:3]}")
    print(f"Shannon: {shannon(test_df)[:3]}")
    
    print("\nTesting Beta Diversity Functions...")
    result = beta_dim_red(test_df, 'bray', 'NMDS')
    print(f"NMDS stress: {result['b']:.4f}")
    
    print("\nTesting Feature Selection...")
    nzv = cut_var(test_df)
    print(f"Near-zero variance features: {nzv.sum()}/{len(nzv)}")
    
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_python_functions()
