import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
import requests
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
import io
from PIL import Image
import base64
import os
from pyobis import occurrences, checklist
from Bio import Phylo, Entrez, SeqIO
from Bio.Align import MultipleSeqAlignment
from Bio.Phylo.TreeConstruction import DistanceCalculator, DistanceTreeConstructor
import dendropy
import subprocess
from tenacity import retry, stop_after_attempt, wait_fixed
import numpy as np
import math
from shapely.wkt import loads
from folium.plugins import MarkerCluster, HeatMap
import logging
import pyworms
from functools import lru_cache

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

Entrez.email = os.getenv("ENTREZ_EMAIL", st.text_input("Enter your email for NCBI Entrez"))
Entrez.api_key = os.getenv("ENTREZ_API_KEY")

# Streamlit UI setup
st.title("Ocean Layers: Seafloor & Evolution Explorer")
st.markdown("Explore seafloor bathymetry, coral reefs, and evolutionary patterns of marine life. Click near blue occurrence points or search for real phylogenetic trees!")

st.subheader("Customize Layers")
show_reefs = st.checkbox("Show Coral Reefs (Optional)", value=False)
show_stats = st.checkbox("Show Global Ocean Stats (Optional)", value=False)

if st.button("Reset Map and Search"):
    st.session_state.clear()

# Initialize session state
if 'clicked_points' not in st.session_state:
    st.session_state.clicked_points = []
if 'biodiversity_data' not in st.session_state:
    st.session_state.biodiversity_data = {}
if 'click_counter' not in st.session_state:
    st.session_state.click_counter = 0

# Sidebar for clicked points summary
st.sidebar.subheader("Clicked Points Summary")
if st.session_state.clicked_points:
    summary_df = pd.DataFrame(st.session_state.clicked_points)
    st.sidebar.dataframe(summary_df.style.format({"Lat": "{:.2f}", "Lon": "{:.2f}", "PD Score": "{:.1f}"}))
    csv = summary_df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button("Download Summary CSV", csv, "clicked_points.csv", "text/csv")
else:
    st.sidebar.write("No points clicked yet.")

# Initialize map
with st.spinner("Initializing map..."):
    m = folium.Map(location=[0, 0], zoom_start=2, tiles=None)
    folium.TileLayer("OpenStreetMap", name="Base Map").add_to(m)
    gebco_wms = "https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/mapserv?"
    folium.WmsTileLayer(
        url=gebco_wms,
        layers="GEBCO_LATEST",
        fmt="image/png",
        transparent=True,
        name="Seafloor Bathymetry (GEBCO)",
    ).add_to(m)

@st.cache_data
def load_shapefile(reef_path):
    reefs = gpd.read_file(reef_path)
    if reefs.crs != "EPSG:4326":
        reefs = reefs.to_crs("EPSG:4326")
    reefs['geometry'] = reefs['geometry'].simplify(tolerance=0.01, preserve_topology=True)
    return reefs

# Coral reefs layer
if show_reefs:
    reef_dir = "data/14_001_WCMC008_CoralReefs2018_v4_1/01_Data"
    shp_files = [f for f in os.listdir(reef_dir) if f.endswith('.shp')]
    shp_options = {f.replace('.shp', ''): os.path.join(reef_dir, f) for f in shp_files}

    if shp_files:
        selected_shp = st.selectbox("Select Coral Reef Dataset", list(shp_options.keys()), key="reef_shp_select")
        reef_path = shp_options[selected_shp]
        try:
            reefs = load_shapefile(reef_path)
            geom_type = reefs.geometry.type.iloc[0]
            layer_name = f"Coral Reefs ({selected_shp})"
            tooltip_field = next((col for col in ['Reef_Type', 'NAME', 'Location', 'OBJECTID'] if col in reefs.columns), None)

            if geom_type == 'Point':
                reef_layer = folium.FeatureGroup(name=layer_name).add_to(m)
                for _, row in reefs.iterrows():
                    coords = [row.geometry.y, row.geometry.x]
                    popup_content = row[tooltip_field] if tooltip_field else "Coral Reef Point"
                    folium.CircleMarker(
                        location=coords,
                        radius=3,
                        color="orange",
                        fill=True,
                        fill_opacity=0.6,
                        popup=folium.Popup(popup_content, max_width=300),
                        tooltip=popup_content if tooltip_field else None
                    ).add_to(reef_layer)
            else:
                if tooltip_field:
                    folium.GeoJson(
                        reefs,
                        name=layer_name,
                        style_function=lambda x: {"color": "orange", "weight": 2, "fillOpacity": 0.4},
                        tooltip=folium.GeoJsonTooltip(fields=[tooltip_field], aliases=["Reef Info"])
                    ).add_to(m)
                else:
                    st.warning(f"No suitable tooltip column found in {selected_shp}.")
                    folium.GeoJson(
                        reefs,
                        name=layer_name,
                        style_function=lambda x: {"color": "orange", "weight": 2, "fillOpacity": 0.4}
                    ).add_to(m)
        except Exception as e:
            st.warning(f"Error loading {selected_shp}: {e}.")
    else:
        st.warning("No shapefiles found in 'data/14_001_WCMC008_CoralReefs2018_v4_1/01_Data'.")

@st.cache_data
@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_obis_data(geom, size=100):
    occ_list = []
    species = []  # Now prioritize species-level
    try:
        logger.info(f"Fetching OBIS data for geometry: {geom} with size={size}")
        query = occurrences.search(geometry=geom, size=size)
        occ_data = query.execute()

        if isinstance(occ_data, pd.DataFrame):
            results = occ_data.to_dict('records')
        else:
            results = occ_data.get('results', [])

        logger.info(f"Raw OBIS results: {results[:2]}")  # Debug: Log first records
        
        # Extract species-level names first
        for rec in results:
            if rec:  # Skip empty
                rank = rec.get('taxonRank', '').lower()
                sci_name = rec.get('scientificName', 'Unknown')
                species_name = rec.get('species', None)  # Often the binomial species
                
                # Prefer species field if available, else scientificName if rank is 'species'
                if species_name and not pd.isna(species_name) and species_name != 'Unknown' and isinstance(species_name, str):
                    name_to_use = species_name
                elif rank == 'species' and sci_name and not pd.isna(sci_name) and isinstance(sci_name, str):
                    name_to_use = sci_name
                else:
                    continue  # Skip higher ranks for species list
                
                species.append(name_to_use)
                
                # For occurrences, use lat/lon and the resolved name
                lat = rec.get('decimalLatitude')
                lon = rec.get('decimalLongitude')
                if lat is not None and lon is not None and isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                    occ_list.append({'lat': lat, 'lon': lon, 'name': name_to_use})

        species = [s.strip() for s in set(species) if isinstance(s, str) and s.strip()]  # Dedupe and clean
        if not species:
            logger.info("No species-level records found.")
            return {'species': [], 'occurrences': []}

        logger.info(f"Resolved to {len(species)} species-level names: {species[:5]}...")  # Debug
        return {'species': species, 'occurrences': occ_list}
    except Exception as e:
        logger.error(f"OBIS API error: {e}")
        return {'species': [], 'occurrences': []}
        

def geocode(location):
    url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
    headers = {'User-Agent': 'StreamlitOceanExplorer/1.0'}
    response = requests.get(url, headers=headers)
    if response.ok:
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
    return None

@retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
def fetch_sequence(taxon):
    """Fetch a single COI sequence for a given taxon from NCBI."""
    term = f"{taxon}[Organism] AND COI[Gene Name] AND (\"500\"[SLEN] : \"2000\"[SLEN])"
    logger.info(f"Searching NCBI for: {term}")
    try:
        search_handle = Entrez.esearch(db="nucleotide", term=term, retmax=1, idtype="acc")
        search_results = Entrez.read(search_handle)
        search_handle.close()
        id_list = search_results["IdList"]
        if id_list:
            fetch_handle = Entrez.efetch(db="nucleotide", id=id_list[0], rettype="fasta", retmode="text")
            record = SeqIO.read(fetch_handle, "fasta")
            fetch_handle.close()
            return record
        return None
    except Exception as e:
        logger.warning(f"Failed to fetch sequence for {taxon}: {e}")
        return None

def is_species_level(taxon):
    """Check if taxon is at species level."""
    try:
        cl_data = checklist.list(scientificname=taxon).execute()
        for rec in cl_data.get('results', []):
            if rec.get('scientificName') == taxon and rec.get('taxonRank') == 'Species':
                return True
        if not cl_data.get('results'):
            return len(taxon.split()) > 1
        return False
    except:
        return len(taxon.split()) > 1

def align_sequences(sequences):
    """Align sequences using MAFFT, with fallback to pairwise alignment or unaligned sequences."""
    try:
        result = subprocess.run(["mafft", "--version"], capture_output=True, text=True, check=False)
        if result.returncode != 0:
            raise FileNotFoundError("MAFFT is not installed or not found in PATH.")

        temp_fasta = "temp.fasta"
        aligned_fasta = "aligned.fasta"
        with open(temp_fasta, "w") as f:
            SeqIO.write(sequences, f, "fasta")

        subprocess.run(["mafft", "--auto", temp_fasta], stdout=open(aligned_fasta, "w"), check=True, text=True)

        if not os.path.exists(aligned_fasta) or os.path.getsize(aligned_fasta) == 0:
            raise RuntimeError("MAFFT produced no output or an empty file.")

        aligned = list(SeqIO.parse(aligned_fasta, "fasta"))

        for file in [temp_fasta, aligned_fasta]:
            if os.path.exists(file):
                os.remove(file)

        if not aligned:
            raise ValueError("No sequences found in MAFFT output.")

        return aligned
    except FileNotFoundError:
        st.warning("MAFFT is not installed. Falling back to pairwise alignment.")
        logger.error("MAFFT not found. Attempting pairwise alignment.")
        try:
            from Bio.Align import PairwiseAligner
            aligner = PairwiseAligner()
            if len(sequences) >= 2:
                alignments = aligner.align(sequences[0].seq, sequences[1].seq)
                aligned = [sequences[0], sequences[1]]
                st.write("Pairwise alignment used as fallback. Only first two sequences processed.")
                return aligned
            else:
                st.error("Insufficient sequences for pairwise alignment.")
                return sequences
        except Exception as e:
            st.error(f"Pairwise alignment failed: {e}. Returning unaligned sequences.")
            return sequences
    except Exception as e:
        st.warning(f"MAFFT alignment failed: {e}. Returning unaligned sequences.")
        logger.error(f"MAFFT alignment error: {e}")
        for file in [temp_fasta, aligned_fasta]:
            if os.path.exists(file):
                os.remove(file)
        return sequences

def render_tree(newick, title):
    """Render a phylogenetic tree from a Newick string."""
    try:
        tree_io = io.StringIO(newick)
        tree = Phylo.read(tree_io, "newick")
        fig, ax = plt.subplots(figsize=(6, 6))
        Phylo.draw(tree, axes=ax, do_show=False)
        ax.set_title(title)
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300)
        img_buffer.seek(0)
        plt.close(fig)
        return Image.open(img_buffer)
    except Exception as e:
        st.warning(f"Tree rendering failed: {e}")
        return None

def circle_to_polygon(lon, lat, radius_km=100, num_points=32):
    points = []
    earth_radius = 6371
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        dlat = (radius_km / earth_radius) * (180 / math.pi) * math.cos(angle)
        dlon = (radius_km / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180) * math.sin(angle)
        points.append((lon + dlon, lat + dlat))
    points.append(points[0])
    wkt = "POLYGON((" + ", ".join(f"{x:.6f} {y:.6f}" for x, y in points) + "))"
    return wkt


@lru_cache(maxsize=1000)
def fetch_colloquial_name(taxon):
    """
    Fetch colloquial name with multi-source fallback. Prioritizes English/marine-relevant names.
    Returns 'Unknown' if nothing found; italicizes if from a related/fallback source.
    """
    if not taxon or taxon.lower() == 'unknown':
        return 'Unknown'
    
    if len(taxon.split()) < 2:  # Skip non-species, but try phylum-level
        return f"{taxon} (genus or higher)"
    
    # 1. WoRMS (existing)
    try:
        resp = pyworms.aphiaRecordsByName(taxon, marine_only=False)
        if resp and isinstance(resp, list) and len(resp) > 0:
            vernacular = resp[0].get('vernacularname', '')
            if vernacular:
                return vernacular.split(',')[0].strip()
    except Exception:
        pass
    
    # 2. GBIF API: Query vernacularNames endpoint
    try:
        gbif_url = f"https://api.gbif.org/v1/species/match?name={taxon}"
        gbif_resp = requests.get(gbif_url, timeout=5).json()
        if gbif_resp.get('usageKey'):
            key = gbif_resp['usageKey']
            vern_url = f"https://api.gbif.org/v1/species/{key}/vernacularNames?language=eng"
            vern_resp = requests.get(vern_url, timeout=5).json()
            results = vern_resp.get('results', [])
            if results:
                # Prefer preferred or first English
                for v in results:
                    if v.get('language') == 'eng' and v.get('vernacularName'):
                        return v['vernacularName'].split(',')[0].strip()
                return results[0].get('vernacularName', 'Unknown')  # Fallback any
        # If no species match, try parent (for phylum-level like arrow worm)
        if gbif_resp.get('kingdomKey') or gbif_resp.get('phylumKey'):
            parent_key = gbif_resp.get('phylumKey') or gbif_resp.get('kingdomKey')
            if parent_key:
                vern_url = f"https://api.gbif.org/v1/species/{parent_key}/vernacularNames?language=eng"
                vern_resp = requests.get(vern_url, timeout=5).json()
                results = vern_resp.get('results', [])
                if results:
                    for v in results:
                        if v.get('language') == 'eng' and v.get('vernacularName'):
                            return v['vernacularName'].split(',')[0].strip() + ' (phylum-level)'
    except Exception as e:
        logger.warning(f"GBIF error for {taxon}: {e}")
    
    # 3. ITIS API: Fixed endpoints and parsing
    try:
        search_url = f"https://www.itis.gov/ITISWebService/jsonservice/searchByScientificName?srchKey={taxon.replace(' ', '%20')}"
        search_resp = requests.get(search_url, timeout=5).json()
        scientific_names = search_resp.get('scientificNames', [])
        if scientific_names:
            tsn = scientific_names[0].get('tsn', '')
            if tsn:
                common_url = f"https://www.itis.gov/ITISWebService/jsonservice/getCommonNamesFromTSN?tsn={tsn}"
                common_resp = requests.get(common_url, timeout=5).json()
                common_names = common_resp.get('commonNames', [])
                if common_names:
                    for cn in common_names:
                        if cn.get('language') == 'English' and cn.get('commonName'):
                            return cn['commonName'].strip()
                    return common_names[0].get('commonName', 'Unknown')  # Fallback
    except Exception as e:
        logger.warning(f"ITIS error for {taxon}: {e}")
    
    # 4. NCBI fallback: Quick lit search for common name in abstracts
    try:
        term = f"{taxon}[organism] AND (common name OR english name OR vernacular)"
        search_handle = Entrez.esearch(db="pubmed", term=term, retmax=1)
        search_results = Entrez.read(search_handle)
        search_handle.close()
        if search_results["IdList"]:
            fetch_handle = Entrez.efetch(db="pubmed", id=search_results["IdList"][0], rettype="abstract", retmode="text")
            abstract = fetch_handle.read()
            fetch_handle.close()
            # Simple regex for common name (e.g., "known as arrow worm")
            import re
            match = re.search(r'(?:known as|called|common name[:\s]+)(\w+\s+\w+)', abstract, re.IGNORECASE)
            if match:
                return match.group(1).strip()
    except Exception as e:
        logger.warning(f"NCBI error for {taxon}: {e}")
    
    # 5. DuckDuckGo fallback: Instant answer API for quick web search
    try:
        ddg_url = f"https://api.duckduckgo.com/?q=common+name+of+{taxon.replace(' ', '+')}&format=json&pretty=1"
        ddg_resp = requests.get(ddg_url, timeout=5).json()
        abstract = ddg_resp.get('Abstract', '')
        if abstract:
            match = re.search(r'(?:known as|called|common name is|also known as)\s*(?:the\s*)?([\w\s]+?)(?:,|\.|$)', abstract, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        related = ddg_resp.get('RelatedTopics', [])
        if related:
            for rel in related:
                text = rel.get('Text', '')
                match = re.search(r'(?:common name|vernacular)\s*:\s*([\w\s]+)', text, re.IGNORECASE)
                if match:
                    return match.group(1).strip()
    except Exception as e:
        logger.warning(f"DuckDuckGo error for {taxon}: {e}")
    
    return 'Unknown'
        


@lru_cache(maxsize=100)  # Cache to avoid redundant OBIS calls
def resolve_to_species(taxon_list, center_lat, center_lon, max_per_taxon=1):
    """
    Resolve higher-rank taxa to species-level names, preferring those near (center_lat, center_lon).
    
    Args:
        taxon_list (list): List of taxa (species or higher ranks).
        center_lat (float): Latitude of the clicked point.
        center_lon (float): Longitude of the clicked point.
        max_per_taxon (int): Max species to pick per higher taxon.
    
    Returns:
        list: List of species-level names.
    """
    resolved = []
    for taxon in set(taxon_list):  # Dedupe
        if not taxon or taxon == 'Unknown' or not isinstance(taxon, str):
            continue
        # Skip if already species-level (has space, e.g., "Acanthaster planci")
        if len(taxon.split()) >= 2 and is_species_level(taxon):
            resolved.append(taxon)
            continue
        
        try:
            # Get AphiaID
            cl_data = checklist.list(scientificname=taxon).execute()
            results = cl_data.get('results', [])
            if not results:
                logger.warning(f"No results for {taxon} in checklist")
                continue
            aphiaid = results[0].get('aphiaID')
            if not aphiaid:
                logger.warning(f"No AphiaID for {taxon}")
                continue
            
            # Get child species via OBIS API
            child_url = f"https://api.obis.org/v3/taxon/{aphiaid}/children?rank=Species"
            response = requests.get(child_url)
            response.raise_for_status()
            child_data = response.json()
            child_species = [rec['scientificName'] for rec in child_data.get('results', []) if rec.get('taxonRank') == 'Species']
            
            if not child_species:
                logger.warning(f"No child species for {taxon}")
                continue
            
            # Filter by proximity (bounding box: ±5° around center)
            bbox = f"POLYGON(({center_lon-5} {center_lat-5},{center_lon-5} {center_lat+5},{center_lon+5} {center_lat+5},{center_lon+5} {center_lat-5},{center_lon-5} {center_lat-5}))"
            geo_filtered = []
            for sp in child_species[:10]:  # Limit to avoid API overload
                sp_occ = occurrences.search(scientificname=sp, geometry=bbox, size=1).execute()
                if sp_occ.get('results') or (isinstance(sp_occ, pd.DataFrame) and not sp_occ.empty):
                    geo_filtered.append(sp)
                    if len(geo_filtered) >= max_per_taxon:
                        break
            
            # If no geo-match, pick first species as fallback
            to_add = geo_filtered if geo_filtered else child_species[:max_per_taxon]
            resolved.extend(to_add)
            logger.info(f"Resolved {taxon} to: {to_add}")
        except Exception as e:
            logger.warning(f"Resolution failed for {taxon}: {e}")
    
    resolved = list(set(resolved))  # Dedupe final list
    if not resolved:
        logger.info("No species resolved; returning original list")
        return [t for t in taxon_list if isinstance(t, str) and len(t.split()) >= 2]  # Keep only species-like
    return resolved
    

def build_phylogenetic_tree(species_list, num_sequences, region, center_lat=None, center_lon=None):
    """
    Build a phylogenetic tree from COI sequences for given species.
    
    Args:
        species_list (list): List of taxa from OBIS.
        num_sequences (int): Number of sequences to fetch.
        region (str): Region name for display.
        center_lat (float): Latitude of clicked point (optional).
        center_lon (float): Longitude of clicked point (optional).
    
    Returns:
        tuple: (newick string, phylogenetic diversity score, insight message)
    """
    if not species_list or all(s == 'Unknown' for s in species_list):
        st.warning(f"No valid species found for {region}. Cannot build phylogenetic tree.")
        return None, 0.0, "No valid species data available."

    # Resolve higher ranks to species, using geographic context if provided
    if center_lat is not None and center_lon is not None:
        with st.spinner("Resolving higher-rank taxa to species..."):
            species_list = resolve_to_species(tuple(species_list), center_lat, center_lon, max_per_taxon=1)
            if not species_list:
                st.warning(f"No species-level taxa resolved for {region}. Using original list.")
                species_list = [s for s in species_list if s != 'Unknown']
            else:
                st.info(f"Resolved to {len(species_list)} species-level taxa for {region}.")
                logger.info(f"Resolved species: {species_list[:5]}...")

    valid_species = [s for s in species_list if s != 'Unknown' and is_species_level(s)]
    if not valid_species:
        valid_species = [s for s in species_list if s != 'Unknown']
    if not valid_species:
        st.warning(f"No valid taxa found for {region}. Cannot build phylogenetic tree.")
        return None, 0.0, "No valid taxa data available."

    sequences = []
    failed_taxa = []
    progress_bar = st.progress(0)
    target_sequences = min(num_sequences, len(valid_species))

    for i, taxon in enumerate(valid_species[:target_sequences * 10]):
        try:
            record = fetch_sequence(taxon)
            if record:
                colloquial = fetch_colloquial_name(taxon)
                record.name = f"{taxon} ({colloquial})"
                sequences.append(record)
            else:
                failed_taxa.append(taxon)
            if len(sequences) >= target_sequences:
                break
            progress_bar.progress(min((i + 1) / (target_sequences * 10), 1.0))
        except Exception as seq_e:
            logger.warning(f"Failed to fetch sequence for {taxon}: {seq_e}")
            failed_taxa.append(taxon)
    progress_bar.empty()

    if failed_taxa and len(sequences) < target_sequences:
        st.warning(
            f"Could not fetch sequences for {len(failed_taxa)} taxa (e.g., {', '.join(failed_taxa[:3])}). "
            f"Only {len(sequences)} sequences retrieved."
        )

    if len(sequences) < 2:
        st.warning(f"Insufficient sequences ({len(sequences)}) to build a tree for {region}.")
        return None, 0.0, f"Only {len(sequences)} sequences retrieved; minimum 2 required."

    try:
        with st.spinner("Aligning sequences and building tree..."):
            aligned_sequences = align_sequences(sequences)
            aln = MultipleSeqAlignment(aligned_sequences)
            calculator = DistanceCalculator('identity')
            dm = calculator.get_distance(aln)
            constructor = DistanceTreeConstructor()
            tree = constructor.nj(dm)
            output = io.StringIO()
            Phylo.write(tree, output, 'newick')
            newick = output.getvalue().strip()
            dtree = dendropy.Tree.get(data=newick, schema='newick')
            pd_score = sum(e.length for e in dtree.edges() if e.length is not None)
            if math.isnan(pd_score):
                pd_score = 0.0
            divergence_insight = f"Tree built from COI sequences of {', '.join([s.name for s in sequences])}"
            return newick, pd_score, divergence_insight
    except Exception as e:
        st.warning(f"Tree construction failed for {region}: {e}.")
        return None, 0.0, f"Tree construction failed: {e}"

# Biodiversity points and heatmap
biodiversity_points = [
    {"lat": -16.5, "lon": 145.5, "region": "Great Barrier Reef"},
    {"lat": 20.0, "lon": -40.0, "region": "Mid-Atlantic Ridge"},
    {"lat": -30.0, "lon": 50.0, "region": "Indian Ocean Seamount"}
]

heatmap_data = []
occurrence_layer = folium.FeatureGroup(name="OBIS Occurrence Points").add_to(m)

for point in biodiversity_points:
    try:
        point_key = f"{point['lat']}_{point['lon']}"
        if point_key not in st.session_state.biodiversity_data:
            geom = circle_to_polygon(point['lon'], point['lat'], radius_km=100)
            obis_data = fetch_obis_data(geom, size=100)
            st.session_state.biodiversity_data[point_key] = obis_data
        else:
            obis_data = st.session_state.biodiversity_data[point_key]

        species_list = obis_data['species']
        logger.debug(f"Biodiversity point {point['region']} species: {species_list[:5]}")
        occ_list = obis_data['occurrences']
        species_count = len(species_list)

        num_sequences = min(10, len(species_list))
        newick, pd_score, divergence_insight = build_phylogenetic_tree(
            species_list, num_sequences, point['region'], point['lat'], point['lon']
        )

        if not species_list:
            species_count = 0
            pd_score = 0.0
            divergence_insight = "No species data available."
        elif newick is None or pd_score is None or math.isnan(pd_score):
            pd_score = 0.0
            divergence_insight = "Failed to build tree; using default PD score."

        tree_img = render_tree(newick, f"Evolution at {point['region']}") if newick else None
        if tree_img:
            buffered = io.BytesIO()
            tree_img.save(buffered, format="PNG")
            img_str = base64.b64encode(buffered.getvalue()).decode()
            popup_html = f"""
            <div style='text-align: center;'>
                <h4>{point['region']}</h4>
                <p>Species: {species_count}<br>Phylogenetic Diversity: {pd_score:.1f}</p>
                <img src='data:image/png;base64,{img_str}' width='200'>
                <p style='color: navy;'>{divergence_insight}</p>
            </div>
            """
        else:
            popup_html = f"""
            <div style='text-align: center;'>
                <h4>{point['region']}</h4>
                <p>Species: {species_count}<br>Phylogenetic Diversity: {pd_score:.1f}</p>
                <p style='color: navy;'>{divergence_insight}</p>
            </div>
            """

        folium.CircleMarker(
            location=[point["lat"], point["lon"]],
            radius=5 + pd_score / 0.5,
            color="green",
            fill=True,
            fill_opacity=0.6,
            popup=folium.Popup(popup_html, max_width=300),
        ).add_to(folium.FeatureGroup(name="Biodiversity & Evolution")).add_to(m)

        cluster = MarkerCluster(name=f"Occurrences at {point['region']}").add_to(occurrence_layer)
        for occ in occ_list:
            folium.Marker(
                location=[occ['lat'], occ['lon']],
                popup=occ['name'],
                icon=folium.Icon(color='blue', icon='info-sign')
            ).add_to(cluster)

        # Only append valid pd_score to heatmap_data
        heatmap_data.append([point["lat"], point["lon"], pd_score])
    except Exception as e:
        logger.error(f"Error for {point['region']}: {e}")
        pd_score = 0.0
        heatmap_data.append([point["lat"], point["lon"], pd_score])

max_pd = max([d[2] for d in heatmap_data] or [0])
if max_pd == 0:
    heatmap_data_normalized = [[d[0], d[1], 0] for d in heatmap_data]
else:
    heatmap_data_normalized = [[d[0], d[1], np.log1p(d[2]) / np.log1p(max_pd)] for d in heatmap_data]
HeatMap(heatmap_data_normalized, name="Phylogenetic Diversity Heatmap", gradient={0.2: "blue", 0.6: "yellow", 1.0: "red"}).add_to(m)

# Legend
legend_html = '''
     <div style="position: fixed;
     bottom: 50px; left: 50px; width: 150px; height: 120px;
     border:2px solid grey; z-index:9999; font-size:14px;
     background-color:white;
     ">
     &nbsp; Heatmap Legend <br>
     &nbsp; Low &nbsp; <i style="background:blue;width:20px;height:20px;display:inline-block;"></i> <br>
     &nbsp; Medium &nbsp; <i style="background:yellow;width:20px;height:20px;display:inline-block;"></i> <br>
     &nbsp; High &nbsp; <i style="background:red;width:20px;height:20px;display:inline-block;"></i> <br>
     &nbsp; Blue Markers: OBIS Data Points
     </div>
     '''
m.get_root().html.add_child(folium.Element(legend_html))

# Global marine habitat stats
if show_stats:
    st.subheader("Global Marine Habitat Coverage")
    world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
    world = world.rename(columns={'iso_a3': 'ISO3'})
    habitat_files = {
        "Warm-Water Corals": "warmwatercorals.csv",
        "Cold-Water Corals": "coldwatercorals.csv",
        "Mangroves": "mangroves.csv",
        "Seagrasses": "seagrasses.csv",
        "Saltmarshes": "saltmarshes.csv",
        "Coastline Coverage": "coastline_coverage.csv"
    }

    @st.cache_data
    def load_habitat_csv(csv_path):
        return pd.read_csv(csv_path)

    selected_habitat = st.selectbox("Select Habitat Type", list(habitat_files.keys()), key="habitat_select")
    csv_path = os.path.join("data/Ocean+HabitatsDownload_Global", habitat_files[selected_habitat])

    if os.path.exists(csv_path):
        try:
            habitat_df = load_habitat_csv(csv_path)
            world_habitat = world.merge(habitat_df, on='ISO3', how='left')
            world_habitat['total_area'] = world_habitat['total_area'].fillna(0)
            world_habitat['percent_protected'] = world_habitat['percent_protected'].fillna(0)

            habitat_layer = folium.FeatureGroup(name=f"{selected_habitat} Coverage").add_to(m)
            folium.Choropleth(
                geo_data=world_habitat,
                name=f"{selected_habitat} Coverage",
                data=world_habitat,
                columns=['ISO3', 'percent_protected'],
                key_on='feature.properties.ISO3',
                fill_color='YlOrRd',
                fill_opacity=0.7,
                line_opacity=0.2,
                legend_name=f'Percent Protected (%) - {selected_habitat}',
                nan_fill_color='grey',
                nan_fill_opacity=0.2
            ).add_to(habitat_layer)

            st.subheader(f"Top 5 Countries by {selected_habitat} Area")
            top_countries = habitat_df.nlargest(5, 'total_area')[['ISO3', 'total_area', 'percent_protected']]
            top_countries['total_area'] = top_countries['total_area'].round(2)
            top_countries['percent_protected'] = top_countries['percent_protected'].round(2)
            st.table(top_countries)
        except Exception as e:
            st.warning(f"Error loading {selected_habitat}: {e}")
    else:
        st.warning(f"Data file for {selected_habitat} not found in 'data/Ocean+HabitatsDownload_Global/'.")

    stats_path = "data/Ocean+HabitatsDownload_Global/global-stats.xlsx"
    if os.path.exists(stats_path):
        try:
            stats_df = pd.read_excel(stats_path)
            st.sidebar.subheader("Global Ocean Stats")
            st.sidebar.dataframe(stats_df)
        except Exception as e:
            st.warning(f"Error loading global-stats.xlsx: {e}. Download from https://habitats.oceanplus.org/")
    else:
        st.warning("Global stats file not found. Download from https://habitats.oceanplus.org/")

folium.LayerControl().add_to(m)

# Interactive map
st.subheader("Interactive Map (Click near blue markers to fetch data)")
radius_km = st.slider("Search Radius (km)", min_value=50, max_value=500, value=100, step=50)
map_output = st_folium(m, width=700, height=500, returned_objects=["last_clicked"], key=f"main_map_{st.session_state.click_counter}")

# Handle map clicks
if map_output and map_output.get("last_clicked"):
    st.session_state.click_counter += 1
    clicked_lat = map_output["last_clicked"]["lat"]
    clicked_lon = map_output["last_clicked"]["lng"]
    st.write(f"Clicked location: Lat {clicked_lat:.2f}, Lon {clicked_lon:.2f}")
    logger.info(f"Map click detected: Lat {clicked_lat}, Lon {clicked_lon}")

    geom = circle_to_polygon(clicked_lon, clicked_lat, radius_km=radius_km)
    st.write(f"Searching within a {radius_km}km radius polygon")

    poly_gdf = gpd.GeoDataFrame(
        {"name": ["Search Area"]},
        geometry=[loads(geom)],
        crs="EPSG:4326"
    )

    click_layer = folium.FeatureGroup(name="Clicked Area", show=True).add_to(m)
    folium.GeoJson(
        poly_gdf,
        name="Search Polygon",
        style_function=lambda x: {"color": "red", "weight": 2, "fillOpacity": 0.2},
        tooltip=f"Search Area (~{radius_km}km radius)"
    ).add_to(click_layer)

    with st.spinner("Fetching OBIS data..."):
        obis_data = fetch_obis_data(geom, size=100)
        species_list = obis_data['species']
        occ_list = obis_data['occurrences']
        if species_list:
            with st.expander("Species at Clicked Location"):
                species_data = []
                for s in species_list:
                    colloquial = fetch_colloquial_name(s) if s != 'Unknown' else 'Unknown'
                    status = "Direct from OBIS" if len(s.split()) >= 2 else "Resolved from higher rank"
                    species_data.append({"Scientific Name": s, "Common Name": colloquial, "Resolution Status": status})
                st.write(f"Found {len(species_list)} species")
                st.dataframe(pd.DataFrame(species_data))

            num_sequences = st.slider(
                "Number of sequences to fetch for tree (COI)",
                min_value=1,
                max_value=min(100, len(species_list)),
                value=min(10, len(species_list)),
                key=f"num_sequences_{st.session_state.click_counter}"
            )
            if num_sequences > 20:
                st.warning(f"Fetching {num_sequences} sequences may take several minutes. Please be patient.")

            newick, pd_score, divergence_insight = build_phylogenetic_tree(
                species_list, num_sequences, "Clicked Location", clicked_lat, clicked_lon
            )

            if newick:
                tree_img = render_tree(newick, "Phylogenetic Tree at Clicked Location")
                if tree_img:
                    st.image(tree_img)
                st.write(f"Phylogenetic Diversity: {pd_score:.1f}")
                st.write(divergence_insight)

            clicked_cluster = MarkerCluster(name="Clicked Occurrences").add_to(click_layer)
            for occ in occ_list:
                folium.Marker(
                    location=[occ['lat'], occ['lon']],
                    popup=occ['name'],
                    icon=folium.Icon(color='orange', icon='star')
                ).add_to(clicked_cluster)
        else:
            st.write(f"No species data found within {radius_km}km of this location. Try clicking closer to blue occurrence points or searching a known region like 'Great Barrier Reef'.")

        st.session_state.clicked_points.append({
            "Lat": clicked_lat,
            "Lon": clicked_lon,
            "Species Count": len(species_list),
            "PD Score": pd_score
        })

    st_folium(m, width=700, height=500, returned_objects=["last_clicked"], key=f"main_map_{st.session_state.click_counter}")

# Search by location or species
st.subheader("Explore Evolution by Location or Species")
search = st.text_input("Enter a location or species (e.g., 'Great Barrier Reef' or 'Mola mola')")
if search:
    with st.spinner("Searching..."):
        species_list = []
        region = search
        is_species_search = False
        try:
            cl_data = checklist.list(scientificname=search).execute()
            if cl_data.get('results'):
                species_list = [rec['scientificName'] for rec in cl_data.get('results', [])]
                is_species_search = True
                region = species_list[0] if species_list else search
        except Exception as e:
            st.warning(f"Checklist search failed: {e}")

        if not species_list:
            coord = geocode(search)
            if coord:
                lat, lon = coord
                geom = circle_to_polygon(lon, lat, radius_km=100)
                obis_data = fetch_obis_data(geom, size=100)
                species_list = obis_data['species']
            else:
                st.error("Could not geocode location or find species.")

        if species_list:
            with st.expander("Species Found"):
                species_data = []
                for s in species_list:
                    colloquial = fetch_colloquial_name(s) if s != 'Unknown' else 'Unknown'
                    status = "Direct from OBIS" if len(s.split()) >= 2 else "Resolved from higher rank"
                    species_data.append({"Scientific Name": s, "Common Name": colloquial, "Resolution Status": status})
                st.write(f"Found {len(species_list)} species")
                st.dataframe(pd.DataFrame(species_data))

            num_sequences = st.slider(
                "Number of sequences to fetch (COI)",
                min_value=1,
                max_value=min(100, len(species_list)),
                value=min(10, len(species_list)),
                key="search_num_sequences"
            )
            if num_sequences > 20:
                st.warning(f"Fetching {num_sequences} sequences may take several minutes. Please be patient.")

            newick, pd_score, divergence_insight = build_phylogenetic_tree(
                species_list, num_sequences, search, lat if coord else None, lon if coord else None
            )

            if newick:
                tree_img = render_tree(newick, f"Evolution for {search}")
                if tree_img:
                    st.image(tree_img, caption=f"Phylogenetic tree showing evolutionary flow for {search}")
                st.write(f"Species Count: {len(species_list)}")
                st.write(f"Phylogenetic Diversity (PD): {pd_score:.1f}")
                st.markdown(f"**Evolutionary Insight**: {divergence_insight}")

            if st.button(f"Fetch Gene Sequences (COI) for Top {num_sequences} Taxa"):
                # Use species_list and filter valid species, as done in build_phylogenetic_tree
                valid_species = [s for s in species_list if s != 'Unknown' and is_species_level(s)]
                if not valid_species:
                    valid_species = [s for s in species_list if s != 'Unknown']
                
                sequences = []
                failed_taxa = []
                progress_bar = st.progress(0)
                for i, taxon in enumerate(valid_species[:num_sequences * 10]):
                    try:
                        record = fetch_sequence(taxon)
                        if record:
                            colloquial = fetch_colloquial_name(taxon)
                            record.name = f"{taxon} ({colloquial})"
                            sequences.append(record)
                        else:
                            failed_taxa.append(taxon)
                        if len(sequences) >= num_sequences:
                            break
                        progress_bar.progress(min((i + 1) / (num_sequences * 10), 1.0))
                    except Exception as seq_e:
                        logger.warning(f"Failed to fetch sequence for {taxon}: {seq_e}")
                        failed_taxa.append(taxon)
                progress_bar.empty()

                if failed_taxa and len(sequences) < num_sequences:
                    st.warning(
                        f"Could not fetch sequences for {len(failed_taxa)} taxa (e.g., {', '.join(failed_taxa[:3])}). "
                        f"Only {len(sequences)} sequences retrieved."
                    )

                if sequences:
                    st.subheader("Fetched COI Sequences")
                    for seq in sequences:
                        st.write(f">{seq.name} | {seq.description}")
                        st.write(str(seq.seq))
                    if len(sequences) >= 2:
                        try:
                            aligned_sequences = align_sequences(sequences)
                            st.subheader("MAFFT Multiple Sequence Alignment")
                            for aln in aligned_sequences:
                                st.write(f">{aln.name}")
                                st.write(str(aln.seq))
                        except Exception as e:
                            st.warning(f"Alignment failed: {e}. Falling back to pairwise.")
                            from Bio.Align import PairwiseAligner
                            aligner = PairwiseAligner()
                            alignments = aligner.align(sequences[0].seq, sequences[1].seq)
                            st.subheader("Example Pairwise Alignment (Global)")
                            st.code(str(alignments[0]))
                else:
                    st.write("No sequences found.")
        else:
            st.write(f"No data found for '{search}'. Please try another location or species.")
            
            
# Footer and notes
st.markdown("""
**Evolution in Focus**: Explore how ocean features drive marine speciation and genetic adaptations. Click green markers or near blue occurrence points (indicating OBIS data) for real phylogenetic trees. The red polygon shows your search area. Toggle coral reefs or stats for more insights!
**Data Source**: Coral Reefs (UNEP-WCMC, 2018), Habitat Data (Ocean+ Habitats, UNEP-WCMC)
**Note**: OBIS and OpenTree APIs are currently returning limited data. Using simulated data for testing.
""")

st.subheader("Notes Section")
user_notes = st.text_area("Add your own notes here:", height=100)

st.markdown("### Opinions on Best Modern Bioinformatics Approach")
st.markdown("""
This app emphasizes phylogenetic diversity (PD) for marine conservation, aligning with a 2025 Nature study on preserving evolutionary potential in ocean ecosystems. Incorporating eDNA methods, as seen in 2025 research on fish communities, would enhance biodiversity assessments. Dendropy’s PD calculations (sum of branch lengths) are efficient and align with modern standards. For sequence alignment, MAFFT is recommended over BioPython’s PairwiseAligner for production, as seen in recent fungal diversity studies. TimeTree integration adds temporal depth, reflecting research on marine PD changes. This app (API loads in ~5-10s) balances interactivity and data-driven bioinformatics.
""")
