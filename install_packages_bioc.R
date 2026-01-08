# install_packages_bioc.R - Simplified for Bioconductor Docker image
# This image already has BiocManager and many packages pre-installed

cat('\n=== R Package Installation for Bioconductor Docker ===\n')
cat('R version:', as.character(getRversion()), '\n')

# Load BiocManager (already installed in bioconductor/bioconductor_docker)
library(BiocManager)
cat('BiocManager version:', as.character(BiocManager::version()), '\n')

# Configure for offline/firewall environments
Sys.setenv(BIOCONDUCTOR_ONLINE_VERSION_DIAGNOSIS='FALSE')

cat('\n--- Installing CRAN packages ---\n')
# Install caret and other CRAN dependencies
cran_packages <- c('caret', 'vegan')
for (pkg in cran_packages) {
    if (!require(pkg, character.only = TRUE, quietly = TRUE)) {
        cat(paste('Installing', pkg, '...\n'))
        install.packages(pkg, repos='http://cran.rstudio.com/', dependencies = TRUE, Ncpus = 2)
        if (require(pkg, character.only = TRUE, quietly = TRUE)) {
            cat(paste('✅', pkg, 'installed\n'))
        }
    } else {
        cat(paste('✅', pkg, 'already installed\n'))
    }
}

cat('\n--- Installing/Updating ANCOMBC and dependencies ---\n')
ancombc_installed <- FALSE

# Strategy 1: Try installing ANCOMBC directly (may already be installed)
cat('Checking if ANCOMBC is already available...\n')
if (require('ANCOMBC', quietly=TRUE)) {
    ancombc_installed <- TRUE
    cat('✅ ANCOMBC already installed\n')
} else {
    cat('Installing ANCOMBC...\n')
    
    # Try installation
    tryCatch({
        BiocManager::install(
            'ANCOMBC',
            ask = FALSE,
            update = FALSE,
            dependencies = TRUE,
            Ncpus = 2
        )
        
        # Verify
        if (require('ANCOMBC', quietly=TRUE)) {
            ancombc_installed <- TRUE
            cat('✅ ANCOMBC installed successfully\n')
        }
    }, error = function(e) {
        cat('❌ ANCOMBC installation failed:', conditionMessage(e), '\n')
    })
}

# Final verification
if (ancombc_installed) {
    cat('\n✅ Installation completed successfully\n')
    cat('Verifying ancombc2 function...\n')
    
    # Check if ancombc2 function exists
    if (exists('ancombc2', where='package:ANCOMBC', mode='function')) {
        cat('✅ ancombc2() function is available\n')
        cat('✅ R ANCOM-BC2 method enabled for publication-quality differential abundance\n')
    } else {
        cat('⚠️ ancombc2() function not found - checking package version\n')
        cat('ANCOMBC version:', as.character(packageVersion('ANCOMBC')), '\n')
    }
} else {
    cat('\n❌ ANCOMBC installation failed\n')
    cat('Application will fall back to Python-based CLR method\n')
    quit(status=1)
}

cat('\n✅ All done!\n')
