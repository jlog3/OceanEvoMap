# OceanEvoMap: Interactive Marine Evolution Visualization

**OceanEvoMap** is an interactive Streamlit application designed to visualize seafloor bathymetry, coral reefs, and evolutionary patterns of marine life. It integrates real data from OBIS, OpenTreeOfLife (OTL), and other sources to provide a dynamic exploration tool for marine biodiversity and phylogenetics.

The app features a global map with layers for GEBCO bathymetry, coral reefs, biodiversity points, and a phylogenetic diversity (PD) heatmap. Users can interact with the map by clicking biodiversity markers or any point to display phylogenetic trees and fetch real-time OBIS data. A search feature enables exploration by region or species, with genetic sequence retrieval and alignment powered by BioPython.

This project showcases advanced geospatial data handling, phylogenetic visualization, and modern bioinformatics integration for evolutionary analysis.

## Features

- **Interactive Map**: Visualize GEBCO bathymetry, coral reefs, biodiversity points, and a PD heatmap.
- **Phylogenetic Trees**: Generate trees from OTL with real PD calculations using Dendropy by clicking markers or map points.
- **Dynamic Data**: Fetch OBIS species data and NCBI gene sequences (e.g., COI) for selected taxa.
- **Search Functionality**: Explore regions (e.g., "Great Barrier Reef") or species (e.g., *Mola mola*) with geocoding and fallback to simulated data.
- **Notes Section**: Add user notes and view Grok's insights on modern bioinformatics approaches.
- **Lightweight Design**: Optimized API calls for ~5-10s load times.

## Prerequisites

Before setting up OceanEvoMap, ensure the following are installed:

1. **Conda**: Install Miniconda from [https://docs.conda.io/en/latest/miniconda.html](https://docs.conda.io/en/latest/miniconda.html).
2. **System Dependencies (Ubuntu)**: Install geospatial libraries:
   ```bash
   sudo apt-get update
   sudo apt-get install -y libgdal-dev gdal-bin

3. **Git**: Ensure Git is installed to clone the repository:
  ```bash
  sudo apt-get install -y git

4. **NCBI Entrez**: Update Entrez.email in app.py with a valid email for API access.

## Setup Instructions
1. **Clone the Repository**:
  ```bash
  git clone https://github.com/jlog3/OceanEvoMap.git
  cd OceanEvoMap
  ```
2. **Create the Conda Environment**:
  ```bash
conda env create -f environment.yml
```
3. **Activate the Environment**:
```bash
conda activate OceanEvoMap
```
4. **Run the App**:
```bash
streamlit run app.py
```
### Alternative Setup with pip
If you prefer pip, install dependencies from requirements.txt:
```bash
pip install -r requirements.txt
```
**Note**: Some packages (e.g., GeoPandas, Dendropy) may require additional system dependencies. Conda is recommended for simplicity. Running the App After setup, run the app:
```bash
streamlit run app.py
```
The app will open in your default browser, displaying an interactive map with seafloor bathymetry, optional coral reefs, and biodiversity markers. Use the search bar to explore regions or species, click on the map to fetch OBIS data, or use the notes section to record observations.

### Project Structure
  	- app.py: Main Streamlit script with map, phylogenetic, and genetic features.
  	- environment.yml: Conda environment file with dependencies.
  	- requirements.txt: Optional pip dependencies.
  	- data/: Folder for local data (e.g., coral reef shapefiles, global stats CSV).
  	- assets/: Folder for images or static files.

Data Sources GEBCO: Bathymetry data via WMS https://www.gebco.net.
OBIS: Species occurrence data https://obis.org.
OpenTreeOfLife: Phylogenetic trees https://opentreeoflife.org.
NCBI: COI gene sequences via Entrez https://www.ncbi.nlm.nih.gov.
TimeTree: Divergence time estimates http://timetree.org.

### Notes on Bioinformatics
OceanEvoMap integrates modern bioinformatics tools like BioPython for sequence alignment and Dendropy for PD calculations, aligning with 2025 research emphasizing PD for conservation (Nature, 2025). For production use, consider:Using MAFFT for multiple sequence alignments.
Integrating eDNA data for enhanced biodiversity insights, as seen in recent studies (e.g., Three Gorges Dam fish communities, 2025).

### License
This project is licensed under the MIT License. See the LICENSE file for details.ContactFor questions or feedback, please open an issue on the GitHub repository: https://github.com/jlog3/OceanEvoMap

### Acknowledgments
Built with Streamlit, Folium, GeoPandas, BioPython, and Dendropy.
Thanks to OBIS, GEBCO, OpenTreeOfLife, NCBI, and TimeTree for providing open data access.
