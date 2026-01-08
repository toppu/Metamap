library(vegan)
library(caret)
library(mia)
library(ANCOMBC)

library ('TreeSummarizedExperiment')

# =============================================================================
# ALPHA DIVERSITY FUNCTIONS
# =============================================================================
simpson <- function(df){
  return(diversity(df,index = "simpson"))
}

shannon <- function(df){
  return(diversity(df,index = "shannon"))
}

# =============================================================================
# BETA DIVERSITY AND DIMENSIONALITY REDUCTION
# =============================================================================
# File-based version for large datasets (avoids pyRserve timeout)
beta_dim_red_from_file <- function(input_path, beta_method, dim_method, output_path, status_path) {
  writeLines("RUNNING", status_path)
  
  tryCatch({
    # Read data
    df <- read.csv(input_path, check.names = FALSE)
    if ('id' %in% colnames(df)) {
      df$id <- NULL
    }
    
    # Compute distance
    if (beta_method == 'bray') {
      dist <- vegdist(df, method = 'bray')
    } else if (beta_method == 'chao') {
      dist <- vegdist(df, method = 'chao')
    } else if (beta_method == 'gower') {
      dist <- vegdist(df, method = 'gower')
    } else {
      stop("Invalid beta_method. Choose 'bray', 'chao', or 'gower'.")
    }
    
    # Dimensionality reduction
    if (dim_method == 'NMDS') {
      # Optimize NMDS for large datasets: fewer tries, autotransform off
      projecs <- metaMDS(dist, k = 2, trymax = 10, autotransform = FALSE, 
                        trace = 0, maxit = 200)
      stress <- projecs$stress
      projecs <- projecs$points
      
      # Save results with consistent column names
      result_df <- as.data.frame(projecs)
      colnames(result_df) <- c('MDS1', 'MDS2')
      result_df$stress <- stress
      write.csv(result_df, output_path, row.names = FALSE)
    } else if (dim_method == 'PCoA') {
      projecs <- cmdscale(dist)
      result_df <- as.data.frame(projecs)
      colnames(result_df) <- c('V1', 'V2')
      write.csv(result_df, output_path, row.names = FALSE)
    } else {
      stop("Invalid dim_method. Choose 'NMDS' or 'PCoA'.")
    }
    
    writeLines("SUCCESS", status_path)
    return(TRUE)
  }, error = function(e) {
    # Try multiple ways to get the error message
    error_msg <- tryCatch(conditionMessage(e), error = function(e2) "")
    if (is.null(error_msg) || error_msg == "") {
      error_msg <- tryCatch(as.character(e), error = function(e2) "")
    }
    if (is.null(error_msg) || error_msg == "") {
      error_msg <- "Unknown error occurred in beta_dim_red_from_file"
    }
    
    # Write to stderr for logging
    cat(paste0("R ERROR in beta_dim_red_from_file: ", error_msg, "\n"), file = stderr())
    
    writeLines(paste0("ERROR: ", error_msg), status_path)
    return(FALSE)
  })
}

# =============================================================================
# FEATURE SELECTION FUNCTIONS
# =============================================================================
# nearZeroVar is from caret package
?metaMDS
cut_var <- function(df, cutoff){
  nzv <- nearZeroVar(df, freqCut = cutoff, saveMetrics= TRUE)
  return(nzv$nzv)
}

# =============================================================================
# DIFFERENTIAL ABUNDANCE ANALYSIS USING ANCOM-BC2
# =============================================================================
# File-based version that reads from CSV files and writes result to CSV (for large datasets)
perform_ancom_from_files <- function(df_path, y_path, tab_path, level, output_path, status_path, prv_cut = 0.05) {
  # Write status file to indicate we're starting
  writeLines("RUNNING", status_path)
  
  tryCatch({
    # Read CSVs without setting row.names initially to avoid duplicates error
    df <- read.csv(df_path, check.names = FALSE)
    y <- read.csv(y_path, check.names = FALSE)
    tab <- read.csv(tab_path, check.names = FALSE)
    
    # Process df: set rownames from taxonomy column
    df <- DataFrame(df)
    rownames(df) <- df$taxonomy
    df$taxonomy <- NULL
    df <- as.matrix(df)
    assays = SimpleList(counts = df)
    
    # Process y: set rownames from id column
    rownames(y) <- y$id
    y$id <- NULL
    smd = DataFrame(y)
    
    # Process tab: set rownames from taxonomy column
    tab <- DataFrame(tab)
    rownames(tab) <- tab$taxonomy
    tab$taxonomy <- NULL
    
    tse = TreeSummarizedExperiment(assays = assays,
                                   colData = smd,
                                   rowData = tab)
    
    # Log dimensions for debugging
    cat(sprintf("TSE dimensions: %d taxa, %d samples\n", nrow(tse), ncol(tse)), file=stderr())
    
    # Check if we have enough data
    if (nrow(tse) < 2) {
      stop("Not enough taxa after filtering (need at least 2)")
    }
    if (ncol(tse) < 3) {
      stop("Not enough samples (need at least 3)")
    }
    
    # Run ANCOM-BC with optimized parameters
    cat("Starting ANCOM-BC computation...\n", file=stderr())
    cat(sprintf("This may take several minutes for %d taxa and %d samples...\n", nrow(tse), ncol(tse)), file=stderr())
    cat(sprintf("Using prv_cut = %.2f for prevalence filtering\n", prv_cut), file=stderr())
    
    out = ancombc2(data = tse, assay_name = "counts",
                   tax_level = level, fix_formula ="bin_var", 
                   prv_cut = prv_cut,  # Use dynamic prevalence cutoff from Python
                   p_adj_method = "holm",  # Faster than Bonferroni
                   n_cl = 1)  # Single-threaded for stability
    
    cat("ANCOM-BC computation completed\n", file=stderr())
    res <- out$res
    
    # Save result to CSV file to avoid pyRserve data transfer limits
    write.csv(res, output_path, row.names = FALSE)
    
    # Write success status
    writeLines("SUCCESS", status_path)
    
    return(TRUE)
  }, error = function(e) {
    # Try multiple ways to get the error message
    error_msg <- tryCatch(conditionMessage(e), error = function(e2) "")
    if (is.null(error_msg) || error_msg == "") {
      error_msg <- tryCatch(as.character(e), error = function(e2) "")
    }
    if (is.null(error_msg) || error_msg == "") {
      error_msg <- "Unknown error occurred in ANCOM-BC"
    }
    
    # Write to stderr for logging
    cat(paste0("R ERROR in perform_ancom_from_files: ", error_msg, "\n"), file = stderr())
    
    writeLines(paste0("ERROR: ", error_msg), status_path)
    return(FALSE)
  })
}


