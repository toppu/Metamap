# Microbiome Analysis Toolbox

A Python-based web application for microbiome data analysis and AI predictions, designed to compare two groups of microbiome datasets (e.g., case vs control, positive vs negative).

## Features

- **Statistical Analysis**: Alpha/Beta diversity, differential abundance (ANCOM-BC), multiple statistical tests
- **Machine Learning**: SVM, Logistic Regression, KNN, XGBoost with SHAP interpretability
- **Ecological Measures**: Shannon/Simpson indices, NMDS, PCoA, multiple distance metrics
- **Interactive Visualizations**: Plotly-based charts for PCA, t-SNE, UMAP, diversity plots

## Quick Start

### Using Docker/Podman

```bash
# Pull and run
podman pull ghcr.io/toppu/metamap:latest
podman run -p 8080:8080 ghcr.io/toppu/metamap:latest

# Or build locally
podman build -t metamap .
podman run -p 8080:8080 metamap
```

### Using Docker Compose

```bash
# Start the application
podman compose up -d --build

# View logs
podman compose logs -f

# Stop
podman compose down
```

Access at: **http://localhost:8080**

## Development

```bash
# Clone repository
git clone https://github.com/toppu/Metamap.git
cd Metamap

# Build and run
podman compose up --build

# With volume mounts for live code changes
podman compose up
```

## Architecture

Single container combining:
- **Python/Streamlit** on port 8080 (web interface)
- **R/Rserve** on port 6311 (ANCOM-BC statistical analysis)
- **Base**: Bioconductor Docker image for R package compatibility

## Environment Variables

```bash
STREAMLIT_SERVER_PORT=8080              # Streamlit port
R_SERVICE_URL=http://127.0.0.1:6311    # R service URL
PYTHONWARNINGS="ignore::DeprecationWarning:pkg_resources"
```

## Data Format

Upload CSV files with:
- **Abundance data**: Taxonomy as first column, samples as subsequent columns
- **Metadata**: Sample IDs matching abundance data, binary grouping variable

See `data/` directory for example files.

## Azure Deployment

```bash
# Build and tag
podman build -t metamap:latest .
podman tag metamap:latest <your-acr>.azurecr.io/metamap:latest

# Push to Azure Container Registry
podman push <your-acr>.azurecr.io/metamap:latest

# Deploy to Azure Container App/Instance
```

## Requirements

- Python 3.12 (see `requirements.txt`)
- R with Bioconductor packages (see `install_packages_bioc.R`)
- 4GB RAM minimum, 8GB recommended

## License

MIT License - See LICENSE file

## Credits

Fork of [ZakariyaTaha/Mcb_Website](https://github.com/ZakariyaTaha/Mcb_Website.git) with improvements:
- Python 3.12 compatibility
- Updated dependencies
- Simplified architecture

```bash
git clone https://github.com/toppu/Metamap.git
cd Metamap
```

### Option 1: Using Pre-built Container (Fastest)

**Pull and run from GitHub Container Registry:**

The published image includes both Python (Streamlit) and R (ANCOM-BC) services in a single container.

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

**For local development** (uses separate containers for Python and R):

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

**Note**: The production image (from GHCR) combines both services into a single container for easier deployment.

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
│   │   ├── Ecological_Diversity.py # Diversity measures + ANCOM-BC
│   │   └── Machine_Learning.py     # ML predictions
│   └── utils/
│       ├── helpers.py              # Core helper functions
│       ├── r_service.py            # R service client
│       └── r_support.R             # R functions for ANCOM-BC
├── data/                            # Sample data files
├── assets/                          # Images and static files
├── requirements.txt                 # Python dependencies
├── install_packages_bioc.R          # R package installation script
├── Dockerfile                       # Combined Python + R container
├── compose.yml                      # Container orchestration
└── run_podman.sh                    # Helper script for Podman
```

## Technology Stack

- **Frontend**: Streamlit
- **Data Processing**: pandas, numpy, scipy
- **Machine Learning**: scikit-learn, XGBoost, SHAP
- **Visualization**: Plotly, matplotlib
- **Statistics**: scipy, statsmodels, R (ANCOM-BC via Rserve)
- **Container**: Podman/Docker with Bioconductor base
- **Language**: Python 3.12 + R 4.4
