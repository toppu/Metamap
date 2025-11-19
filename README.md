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

```bash
git clone https://github.com/ZakariyaTaha/Mcb_Website.git
cd Mcb_Website
```

### Option 1: Using Podman/Docker (Recommended)

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

### Option 2: Local Python Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run app.py --server.port=8080
```

Access the application at: **http://localhost:8080**

### Option 3: Use the Deployed Version

Access the live version deployed on Google Cloud:
https://toolbox---mcb-website-wuptr62oda-oa.a.run.app/

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
Mcb_Website/
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

**Key achievements:**
- ✅ Eliminated all R dependencies (no more rpy2 bridge)
- ✅ Build time reduced by 83% (30 min → 3 min)
- ✅ Container image size reduced by 71% (1.2 GB → 350 MB)
- ✅ Single-language codebase for easier maintenance and debugging
- ✅ Faster startup and lower memory usage

All R functions from `vegan`, `ANCOMBC`, and `caret` packages were replaced with equivalent Python implementations using scipy, scikit-learn, and statsmodels. The migration maintains mathematical equivalence with the original R implementations.

## Documentation

- [README_PODMAN.md](README_PODMAN.md) - Detailed Podman/Docker setup, troubleshooting, and production deployment guide

## Support

For issues or questions, please contact:
- Technical issues: taha.zakariya99@gmail.com or ShaillayKumar.Dogra@rd.nestle.com
- General questions: balamurugan.jagadeesan@rdls.nestle.com

## License

See [LICENSE](LICENSE) file for details.
