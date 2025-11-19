# Microbiome Analysis Toolbox

A Python-based web application for microbiome data analysis and AI predictions, designed to compare two groups of microbiome datasets (e.g., case vs control, positive vs negative).

## Features

### Statistical Analysis
- **Alpha diversity**: Shannon and Simpson diversity indices
- **Beta diversity**: NMDS and PCoA with multiple distance metrics (Bray-Curtis, Jaccard, Euclidean, Gower)
- **Differential abundance**: ANCOM alternative using Mann-Whitney U test with FDR correction
- **Statistical tests**: t-test, Mann-Whitney U, Wilcoxon, Kruskal-Wallis
- **Dimensionality reduction**: PCA, t-SNE, UMAP

### Ecological Measures
- Shannon diversity index for species diversity
- Simpson diversity index for community evenness
- Non-metric multidimensional scaling (NMDS)
- Principal Coordinates Analysis (PCoA)
- Multiple beta diversity metrics

### Machine Learning
- Support Vector Machines (SVM)
- Logistic Regression
- K-Nearest Neighbors (KNN)
- XGBoost gradient boosting
- SHAP feature importance analysis for model interpretability
- Cross-validation and hyperparameter tuning

### Data Processing
- Centered log-ratio (CLR) transformation for compositional data
- Near-zero variance feature filtering
- Phyla ratio calculations
- Taxonomy-based aggregation at multiple levels

### Interactive Visualizations
- Plotly-based interactive charts
- PCA/t-SNE/UMAP scatter plots
- Alpha diversity boxplots
- Beta diversity ordination plots
- Differential abundance bar charts
- Feature importance plots

## Quick Start

### Clone the Repository

This is a fork of [ZakariyaTaha/Mcb_Website](https://github.com/ZakariyaTaha/Mcb_Website.git) with improvements including:
- Removed all R dependencies (pure Python implementation)
- Code refactoring for better maintainability
- Improved build performance and reduced container size

```bash
git clone https://github.com/toppu/Metamap.git
cd Metamap
```

### Option 1: Using Pre-built Container (Fastest)

**Pull and run from GitHub Container Registry:**
```bash
# Using Docker
docker pull ghcr.io/toppu/metamap:latest
docker run -p 8080:8080 ghcr.io/toppu/metamap:latest

# Using Podman
podman pull ghcr.io/toppu/metamap:latest
podman run -p 8080:8080 ghcr.io/toppu/metamap:latest
```

See [CONTAINER.md](CONTAINER.md) for detailed container usage instructions.

### Option 2: Using Podman/Docker with Local Build

**Using the helper script:**
```bash
# Start the application
./run_podman.sh start

# View logs
./run_podman.sh logs

# Stop the application
./run_podman.sh stop
```

**Or manually with Podman:**
```bash
podman-compose up --build
```

**Or with Docker:**
```bash
docker-compose up --build
```

The application will be available at: **http://localhost:8080**

### Option 3: Local Python Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py
```

Access the application at: **http://localhost:8501** (default Streamlit port)

## Data Format Requirements

### Microbiome Data File
- Must have a `taxonomy` column containing taxonomic classifications (e.g., `Kingdom;Phylum;Class;Order;Family;Genus;Species`)
- Additional columns represent sample IDs (must match metadata)
- Supports read counts or relative abundance data

### Metadata File
- Must have an `id` column with sample IDs matching the microbiome data
- Must have a `bin_var` column with exactly 2 groups (e.g., case/control, treated/untreated)
- Groups should be assigned to 0 and 1 for proper analysis

## Project Structure

```
Metamap/
├── app.py                           # Main application entry point
├── src/
│   ├── pages/                       # Streamlit pages
│   │   ├── Welcome.py              # Landing page
│   │   ├── Instructions.py         # Detailed instructions
│   │   ├── Data_Upload.py          # Data upload interface
│   │   ├── Statistical_Analysis.py # Statistical analysis tools
│   │   ├── Ecological_Diversity.py # Diversity measures
│   │   └── Machine_Learning.py     # ML predictions
│   └── utils/
│       ├── helpers.py              # Core helper functions
│       └── python_support.py       # Statistical functions
├── data/                            # Sample data files
├── assets/                          # Images and static files
├── requirements.txt                 # Python dependencies
├── Dockerfile                       # Container definition
└── compose.yml                      # Container orchestration
```

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, XGBoost, SHAP
- **Visualization**: Plotly, matplotlib
- **Statistics**: scipy, statsmodels
- **Container**: Podman/Docker
- **Language**: Pure Python (no R dependencies)

## Migration History

This project was successfully migrated from R+Python (using rpy2) to **pure Python** in November 2025.

**Key changes:**
- Eliminated all R dependencies (no more rpy2 bridge)
- Build time reduced by 83% (30 min → 3 min)
- Container image size reduced by 71% (1.2 GB → 350 MB)
- Single-language codebase for easier maintenance and debugging

All R functions from `vegan`, `ANCOMBC`, and `caret` packages were replaced with equivalent Python implementations using scipy, scikit-learn, and statsmodels. The migration maintains mathematical equivalence with the original R implementations.

## Documentation

- [CONTAINER.md](CONTAINER.md) - Container usage guide (pull, run, tags)
- [README_PODMAN.md](README_PODMAN.md) - Detailed Podman/Docker setup, troubleshooting, and production deployment guide

## Container Registry

Pre-built container images are automatically published to GitHub Container Registry:

- **Latest**: `ghcr.io/toppu/metamap:latest`
- **Versions**: `ghcr.io/toppu/metamap:v1.0.0` (when tagged)
- **Platforms**: `linux/amd64`, `linux/arm64`

Images are automatically built and published on every push to main and when version tags are created.
