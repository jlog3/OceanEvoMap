import streamlit as st
import folium
import geopandas as gpd
import pandas as pd
import requests
from streamlit_folium import st_folium
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
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
from folium.plugins import MarkerCluster, MeasureControl
import pyworms
from functools import lru_cache
import re
import logging # Add this if not present
from Bio.Align import PairwiseAligner # Ensure imported for fallback
from pygbif import occurrences as gbif_occ
import urllib3; urllib3.disable_warnings()
import urllib.parse
from branca.element import MacroElement
from jinja2 import Template
import json
from dotenv import load_dotenv  # Add this if you haven't already, to load your .env file
load_dotenv()  # Call it early in the script

# Temporary patch for pyobis logging bug: Redirect tqdm output to avoid invalid kwargs
class TqdmLoggingHandler(logging.Handler):
    def emit(self, record):
        try:
            msg = self.format(record)
            from tqdm import tqdm # Import here to avoid global
            tqdm.write(msg)
            self.flush()
        except Exception:
            self.handleError(record)

# Add scale
class ScaleControl(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
            L.control.scale({{this.options}}).addTo({{this._parent.get_name()}});
        {% endmacro %}
        """)
    def __init__(self, position='bottomleft', imperial=False, metric=True, **kwargs):
        super().__init__()
        self._name = 'ScaleControl'
        options = {
            'position': position,
            'imperial': imperial,
            'metric': metric
        }
        options.update(kwargs)
        self.options = json.dumps(options)

# Custom AttributionControl class
class AttributionControl(MacroElement):
    _template = Template("""
        {% macro script(this, kwargs) %}
            L.control.attribution({position: 'bottomleft'}).addTo({{this._parent.get_name()}});
        {% endmacro %}
    """)



# In your app setup, after imports:
logger = logging.getLogger('pyobis')
logger.addHandler(TqdmLoggingHandler())
logger.propagate = False # Prevent double-logging
email_default = os.getenv("ENTREZ_EMAIL", "")
Entrez.email = st.text_input("Enter your email for NCBI Entrez", value=email_default)
Entrez.api_key = os.getenv("ENTREZ_API_KEY")

# Streamlit UI setup
st.title("Ocean Layers: Seafloor & Evolution Explorer")
st.markdown("Explore seafloor bathymetry, coral reefs, and evolutionary patterns of marine life. Click near blue occurrence points or search for real phylogenetic trees!")

marine_phyla = [
    "Acanthocephala", "Acoelomorpha", "Annelida", "Arthropoda", "Brachiopoda", "Bryozoa", "Chaetognatha",
    "Chordata", "Cnidaria", "Ctenophora", "Cycliophora", "Echinodermata", "Entoprocta", "Gastrotricha",
    "Gnathostomulida", "Hemichordata", "Kinorhyncha", "Loricifera", "Mollusca", "Nematoda", "Nematomorpha",
    "Nemertea", "Orthonectida", "Phoronida", "Placozoa", "Platyhelminthes", "Porifera", "Priapulida",
    "Rhombozoa", "Rotifera", "Sipuncula", "Tardigrada", "Xenoturbellida"
]

license_links = {
    "CC0": "https://creativecommons.org/publicdomain/zero/1.0/",
    "CC4": "https://creativecommons.org/licenses/by/4.0/",
    "CC3":"https://creativecommons.org/licenses/by-sa/3.0/",
    "CC2": "https://creativecommons.org/licenses/by-sa/2.0/",
    "public domain":"",
    "GNU":"https://commons.wikimedia.org/wiki/GNU_Free_Documentation_License", # GNU Free Documentation License
    "researchgate": ""
}

# source is wikimedia, or author flickr page, (or text about srouce?)
# source_link can be the DOI like 
                #"source": "https://commons.wikimedia.org/wiki/File:SEM_image_of_Milnesium_tardigradum_in_active_state_-_journal.pone.0045682.g001-2.png",
                #"sourcelink": "Schokraie E, Warnken U, Hotz-Wagenblatt A, Grohme MA, Hengherr S, et al. (2012) Comparative proteome analysis of Milnesium tardigradum in early embryonic state versus adults in active and anhydrobiotic state. PLoS ONE 7(9): e45682. doi:10.1371/journal.pone.0045682",
                #"license": "CC2"

# some phyla in your list (e.g., Acoelomorpha and Xenoturbellida) are classified under the phylum Xenacoelomorpha in GBIF's backbone taxonomy—use key '7190138' for both

phylum_info = {
    'Acanthocephala': {
        'description': 'Parasitic worms characterized by an eversible spiny proboscis for attaching to hosts; they lack a digestive tract and absorb nutrients through their body surface, with complex life cycles involving intermediate and definitive hosts.',
        'rarity': 'Rare', # ~10K records
        'habitable_areas': 'Marine and freshwater systems as parasites; often in crustacean intermediate hosts and fish/seabird definitive hosts in coastal and open ocean environments.',
        'representative_species': 'Polymorphus paradoxus (no common name; parasitizes seabirds); Corynosoma wegeneri (no common name; marine parasite)',
        'vernacular': 'Thorny-headed worms',
        'gbif_taxon_id': '67',
        'hotspots': [
            {"lat": 44.65, "lon": -63.57, "region": "Halifax, Canada (Lobster Habitats)"},
            {"lat": 22.63, "lon": 120.27, "region": "Kaohsiung, Taiwan (Red Snapper Habitats)"}
        ],
        'images': [
            {
                "file": "Polymorphus.png",
                "species": "Polymorphus",
                "vernacular": "thorny-headed worm",
                "description": "",
                "attribution": "By Unknown author - https://pubmed.ncbi.nlm.nih.gov/37874424/, CC BY 4.0, https://commons.wikimedia.org/w/index.php?curid=162149367",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Unknown",
                "sourcelink": "",
                "license": "CC4",
            },
            {
                "file": "Hand-drawn_Polymorphus.jpg",
                "species": "Polymorphus",
                "vernacular": "thorny-headed worm",
                "description": "Red pigment depiction of the Polymorphus",
                "attribution": "By Standardusername2 - Own work, CC0, https://commons.wikimedia.org/w/index.php?curid=161786180",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Unknown",
                "sourcelink": "",
                "license": "CC0",
            }
        ]
    },
    'Acoelomorpha': {
        'description': 'Simple, small soft-bodied animals lacking a gut or body cavity, with one opening for digestion/excretion; they are hermaphrodites with a basic nervous system and sensory organs.',
        'habitable_areas': 'Primarily marine or brackish waters, living in sediment grains, as plankton, or on algae/corals; mostly in coastal and shallow marine zones.',
        'rarity': 'Rare',  # <10K records        
        'representative_species': 'Waminoa sp. (no common name; found on corals); Symsagittifera roscoffensis (mint flatworm)',
        'vernacular': 'Acoelomorph flatworms',
        'gbif_taxon_id': '7190138',
        'hotspots': [
            {"lat": 3.25, "lon": 73, "region": "Maldives, Indian Ocean"}
        ],
        'images': [
            {
                "file": "Acoel_Flatworms_(Waminoa_sp.)_on_Bubble_Coral_(Plerogyra_sinuosa)_-_Panglima,_Pulau_Mabul,_Sabah,_Malaysia.jpg",
                "species": "Waminoa sp.",
                "vernacular": "no common name; found on corals",
                "description": "Acoel Flatworms (Waminoa sp.) on Bubble Coral (Plerogyra sinuosa) - Panglima, Pulau Mabul, Sabah, Malaysia",
                "attribution": "",
                "location": "Panglima, Pulau Mabul, Sabah, MALAYSIA",
                "camera_coordinates": "4° 14′ 54.59″ N, 118° 37′ 46.86″ E",
                "date": "28 January 2010",
                "author": "Bernard DUPONT from FRANCE",
                "sourcelink": "https://www.flickr.com/photos/berniedup/6097277270/",
                "license": "CC4",
            },
            {
                "file": "Symsagittifera_roscoffensis(Jersey).jpg",
                "vernacular": "mint flatworm",
                "description": "Symsagittifera roscoffensis (an acoelomorph flatworm, commonly know as the mint sauce worm), in Jersey.",
                "attribution": "By Stevie Smith - https://www.flickr.com/photos/68769579@N07/8479480795/, CC BY 2.0, https://commons.wikimedia.org/w/index.php?curid=35405748",
                "location": None,
                "camera_coordinates": "",
                "date": "April 8, 2008",
                "author": "Stevie Smith",
                "sourcelink": "https://www.flickr.com/photos/68769579@N07/8479480795/",
                "license": "CC2",
            },
            {
                "file": "Symsagittifera_roscoffensis_(from_Graff,_1891).png",
                "vernacular": "mint flatworm",
                "description": "Original painting by Ludwig von Graff",
                "attribution": "By Ludwig fon Graff (1851–1924) - Graff, L. Die Organisation der Turbellaria Acoela mit einem Enhange ueber den Bau und Bedeutung der Chlorophyllzellen von Convoluta roscoffensis von Gotlieb Haberlandt. – Leipzig: Wilhelm Engelmann, 1981. Tafel VII, Public Domain, https://commons.wikimedia.org/w/index.php?curid=35174170",
                "location": None,
                "camera_coordinates": "",
                "date": "1891",
                "author": "Ludwig fon Graff (1851–1924)",
                "sourcelink": "https://www.biodiversitylibrary.org/item/40741#page/125/mode/1up",
                "license": "public domain",
            }
        ]
    },
    'Annelida': {
        'description': 'Bilaterally symmetrical, segmented invertebrates with a coelom, circular segments, and chaetae (bristles) for movement; they have a closed circulatory system and diverse feeding strategies.',
        'habitable_areas': 'Predominantly marine, in tidal zones, hydrothermal vents, coral reefs, and seafloor sediments; polychaetes dominate marine benthic communities.',
        'rarity': 'Common',  # >1M records (e.g., ~800K in GBIF proxy, likely similar in OBIS)        
        'representative_species': 'Nereis (ragworm); Hirudo medicinalis (medicinal leech)',
        'vernacular': 'Segmented worms',
        'gbif_taxon_id': '42',
        'hotspots': [
            {"lat": 9.83, "lon": -104.3, "region": "East Pacific Rise (Hydrothermal Vents)"},
            {"lat": -16.5, "lon": 145.5, "region": "Great Barrier Reef (Coral Reefs)"},
            {"lat": 25.03, "lon": -78.04, "region": "Bahamas, Caribbean (Coral Reefs)"},
            {"lat": 36.8, "lon": -122.0, "region": "Monterey Bay (Tidal Zones)"},
            {"lat": 37.77, "lon": -122.43, "region": "San Francisco Bay (Marine Habitats)"}
        ],
        'images': [
            {
                "file": "Nereididae_(YPM_IZ_035344).jpeg",
                "species": "Nereididae",
                "vernacular": "",
                "description": "Preserved Specimen. Invertebrate",
                "attribution": "Yale Peabody Museum",
                "location": "North America; Atlantic Ocean; Caribbean Sea; British Virgin Islands; Anegada",
                "camera_coordinates": "",
                "date": "2004-05-21",
                "author": "Unknown",
                "sourcelink": "http://collections.peabody.yale.edu/search/Record/YPM-IZ-035344",
                "license": "CC0",
                "license_link": "https://creativecommons.org/licenses/by/4.0/"
            },
            {
                "file": "AmCyc_Leech.jpg",
                "species": "Sanguisuga medicinalis",
                "vernacular": "Leech",
                "description": "1. Leech. 2. Anterior extremity magnified. 3. Jaw detached magnified. 4. Part of belly magnified.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "1879",
                "author": "Unknown",
                "sourcelink": "The American Cyclopædia, v. 10, 1879, p. 310.",
                "license": "public domain",
                "license_link": "https://creativecommons.org/licenses/by/4.0/"
            }
        ]
    },
    'Arthropoda': {
        'description': 'Invertebrates with a chitinous exoskeleton, segmented bodies, jointed appendages, and an open circulatory system; they moult to grow and exhibit high diversity.',
        'habitable_areas': 'Marine ecosystems worldwide, including oceans, deep-sea trenches, and coastal areas; crustaceans are mostly aquatic in marine settings.',
        'rarity': 'Common',  # >10M records        
        'representative_species': 'Macrocheira kaempferi (Japanese spider crab); Homarus americanus (American lobster)',
        'vernacular': 'Arthropods',
        'gbif_taxon_id': '54',
        'hotspots': [
            {"lat": -16.5, "lon": 145.5, "region": "Great Barrier Reef (Crustacean Diversity)"},
            {"lat": 25.03, "lon": -78.04, "region": "Bahamas, Caribbean (Lobster Habitats)"},
            {"lat": -34.93, "lon": 138.6, "region": "Adelaide Coast, South Australia"},
            {"lat": 22.63, "lon": 120.27, "region": "South China Sea (Marine Arthropods)"},
            {"lat": 49.25, "lon": -123.12, "region": "Vancouver Coast, British Columbia"},
            {"lat": -75, "lon": -175, "region": "Ross Sea, Antarctica (Deep Sea)"},
            {"lat": -23.65, "lon": -70.4, "region": "Antofagasta, Chile (Coastal)"}
        ],
        'images': [
            {
                "file": "Macrocheira_kaempferi_01.jpg",
                "species": "Macrocheira kaempferi",
                "vernacular": "Japanese spider crab",
                "description": "Macrocheira kaempferi, Inachidae, Japanese Spider Crab; dissected specimen; Staatliches Museum für Naturkunde Karlsruhe, Germany.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "49° 00′ 23.97″ N, 8° 24′ 01.86″ E",
                "date": "24 October 2010",
                "author": "H. Zell",
                "source": "",
                "sourcelink": "",
                "license": "CC3"
            },
            {
                "file": "640px-Homarus_americanus_(YPM_IZ_103861).jpeg",
                "species": "Homarus americanus",
                "vernacular": "American lobster",
                "description": "PRESERVED_SPECIMEN; ; dry; ; recent gallery; IZ number 103861; lot count 1;",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "2019",
                "author": "Eric A. Lazo-Wasem",
                "source": "Gall L (2019). Invertebrate Zoology Division, Yale Peabody Museum. Yale University Peabody Museum. Occurrence dataset https://doi.org/10.15468/0lkr3w accessed via GBIF.org on 2019-06-22. https://www.gbif.org/occurrence/1986657220",
                "sourcelink": "http://collections.peabody.yale.edu/search/Record/YPM-IZ-103861",
                "license": "CC BY 4.0"
            }
        ]
    },
    'Brachiopoda': {
        'description': 'Marine animals with hinged dorsal-ventral shells and a lophophore for filter-feeding; divided into articulate (toothed hinges) and inarticulate types.',
        'habitable_areas': 'Exclusively marine, in rocky overhangs, caves, continental slopes, and deep ocean floors; prefer cold, low-light waters.',
        'rarity': 'Medium',    # ~50K-100K records      
        'representative_species': 'Lingula anatina (lingulid brachiopod); Terebratalia transversa (no common name)',
        'vernacular': 'Lamp shells',
        'gbif_taxon_id': '110',
        'hotspots': [
            {"lat": 43, "lon": 131, "region": "Sea of Japan"},
            {"lat": -23.65, "lon": -70.4, "region": "Antofagasta, Northern Chile"},
            {"lat": 12.4, "lon": 102.52, "region": "Trat Province, Thailand (Mangroves)"},
            {"lat": -18.14, "lon": 178.44, "region": "Suva, Fiji"},
            {"lat": 35.1, "lon": 139.08, "region": "Japan Coastal"}
        ],
        'images': [
            {
                "file": "Lingula-Photo.jpg",
                "species": "Lingula anatina",
                "vernacular": "lingulid brachiopod",
                "description": "",
                "attribution": "Okinawa Institute of Science and Technology Graduate University",
                "location": "Amami Island, Japan",
                "camera_coordinates": "",
                "date": "18 September 2015",
                "author": "Unknown",
                "source": "OIST (Okinawa Institute of Science and Technology Graduate University)",
                "sourcelink": "https://www.oist.jp/news-center/press-releases/%E2%80%9Cliving-fossil%E2%80%9D-genome-decoded",
                "license": "CC4"
            },
            {
                "file": "Terebratalia_transversa_141510036.jpg",
                "species": "Terebratalia transversa",
                "vernacular": "North Pacific Lampshell",
                "description": "",
                "attribution": "",
                "location": "Strathcona, BC, Canada",
                "camera_coordinates": "49° 56′ 20.51″ N, 125° 10′ 29.28″ W",
                "date": "26 June 2021",
                "author": "Marilynne Box https://www.inaturalist.org/users/2645718",
                "source": "",
                "sourcelink": "https://www.inaturalist.org/photos/141510036",
                "license": "CC4"
            }
        ]
    },
    'Bryozoa': {
        'description': 'Aquatic colonial invertebrates with a lophophore for filter-feeding; colonies consist of interconnected zooids, often encrusting or erect.',
        'habitable_areas': 'Mostly marine in tropical to polar waters, on hard substrates like rocks, shells, and algae; common in coral reefs and deeper seas.',
        'rarity': 'Medium',  # ~200K records        
        'representative_species': 'Membranipora membranacea (marine lace-like bryozoan); Bugula neritina (no common name)',
        'vernacular': 'Moss animals',
        'gbif_taxon_id': '53',
        'hotspots': [
            {"lat": 43, "lon": -69, "region": "Gulf of Maine"},
            {"lat": 47.61, "lon": -122.33, "region": "Seattle, US Northwest Coast"},
            {"lat": 78, "lon": 16, "region": "Svalbard, Arctic"},
            {"lat": -75, "lon": -175, "region": "Ross Sea, Antarctic"},
            {"lat": 20.6, "lon": -16.25, "region": "Banc d'Arguin, Mauritania"},
            {"lat": 35.12, "lon": 33.43, "region": "Cyprus, Mediterranean"},
            {"lat": 35.53, "lon": 129.03, "region": "South Africa Coastal"},
            {"lat": 12.4, "lon": 102.52, "region": "Trat Province, Thailand"}
        ],
        'images': [
            {
                "file": "5cf5df6e5ee161cf03587aba201dcb98.jpg",
                "species": "Triphyllozoon moniliferum",
                "vernacular": "White Lace Coral",
                "description": "White Lace Coral, Triphyllozoon moniliferum, together with purple Lodictyum phoeniceum Image: Dr Isobel Bennett. Introduction: Lace corals are sometimes found washed up on beaches after storms. Identification: This bryozoan forms hardened fan-like lace structures. The colony is attached to rocks and grows in the form of folded lace-like sheets.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Dr Isobel Bennett",
                "source": "Australian Museum",
                "sourcelink": "https://australian.museum/learn/animals/jellyfish/lace-coral/",
                "license": "CC4"
            },
            {
                "file": "Bugula_neritina.jpg",
                "species": "Bugula neritina",
                "vernacular": "",
                "description": "General aspect of a colony of Bugula neritina, a bryozoan",
                "attribution": "By Dean Janiak, Smithsonian Marine Station - Indian River Lagoon Species Inventory. 2023. http://www.irlspecies.org/index.php. Accessed on July 14., CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=134425559",
                "location": None,
                "camera_coordinates": "",
                "date": "14 July 2023",
                "author": "Dean Janiak, Smithsonian Marine Station",
                "source": "Indian River Lagoon Species Inventory. 2023. http://www.irlspecies.org/index.php. Accessed on July 14.",
                "sourcelink": "",
                "license": "CC3"
            }
        ]
    },
    'Chaetognatha': {
        'description': 'Predatory marine worms with a torpedo-shaped body, grasping spines, and fins; they are transparent and planktonic.',
        'habitable_areas': 'All marine waters worldwide, from surface to deep sea and polar regions; primarily pelagic as plankton.',
        'rarity': 'Medium',   # ~50K records       
        'representative_species': 'Sagitta elegans (common arrow worm); Eukrohnia hamata (polar arrow worm)',
        'vernacular': 'Arrow worms',
        'gbif_taxon_id': '55',
        'hotspots': [
            {"lat": 75, "lon": -150, "region": "Canada Basin, Arctic"},
            {"lat": -75, "lon": -175, "region": "Ross Sea, Antarctic"},
            {"lat": -30, "lon": -30, "region": "South Atlantic"},
            {"lat": 35.17, "lon": 129.03, "region": "Busan, Korean Waters"},
            {"lat": 44, "lon": 35, "region": "Black Sea"},
            {"lat": 35, "lon": 30, "region": "Eastern Mediterranean"},
            {"lat": 55, "lon": 3, "region": "North Sea"},
            {"lat": 57, "lon": 20, "region": "Baltic Sea"}
        ],
        'images': [
            {
                "file": "Chaetognatha.png",
                "species": "Chaetognatha",
                "vernacular": "common arrow worm",
                "description": "Chaetognatha and some examples of their diversity; compilation.",
                "attribution": "By Varios autores. Compilación por mí. - Mi compilación de estas imágenes: File:Protosagitta spinosa.jpg, File:Slabber plate 06.jpg, File:Chaetoblack.png (dominio público), CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=29151629",
                "location": None,
                "camera_coordinates": "",
                "date": "23 October 2013",
                "author": "Various",
                "source": "",
                "sourcelink": "https://commons.wikimedia.org/wiki/File:Chaetognatha.PNG",
                "license": "CC3"
            },
            {
                "file": "Arrow-Worms-Chaetognatha-1024x979.jpg",
                "species": "Chaetognatha",
                "vernacular": "polar arrow worm",
                "description": "Arrow worms, also known as chaetognaths, are a group of free-living, predatory marine invertebrates that belong to the phylum Chaetognatha. They have distinct head, trunk, and tail segments, with paired lateral fins and a single caudal fin. Their name originated from their transparent or translucent, arrow-shaped bodies, which range between 1 millimeter to over 12 centimeters. However, species in colder waters are generally larger.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Unknown",
                "source": "Animal Fact",
                "sourcelink": "https://animalfact.com/arrow-worm/",
                "license": "CC4"
            }
        ]
    },
    'Chordata': {
        'description': 'Animals with a notochord, dorsal nerve cord, pharyngeal slits, and post-anal tail at some life stage; includes vertebrates and invertebrate subphyla like tunicates.',
        'habitable_areas': 'Diverse, but marine focus includes seafloor sediments for lancelets and sessile/ pelagic for tunicates in oceans worldwide.',
        'rarity': 'Common',  # >5M records
        'representative_species': 'Branchiostoma lanceolatum (lancelet); Ascidia sp. (sea squirt)',
        'vernacular': 'Chordates',
        'gbif_taxon_id': '44',
        'hotspots': [
            {"lat": 22.63, "lon": 120.27, "region": "South China Sea"},
            {"lat": -16.5, "lon": 145.5, "region": "Great Barrier Reef (Marine Diversity)"},
            {"lat": 30, "lon": -60, "region": "Sargasso Sea (Fish Diversity)"},
            {"lat": -0.5, "lon": -90.5, "region": "Galápagos"},
            {"lat": 25.03, "lon": -78.04, "region": "Bahamas, Caribbean"},
            {"lat": 50, "lon": -30, "region": "North Atlantic"},
            {"lat": -75, "lon": -175, "region": "Ross Sea, Antarctic"},
            {"lat": -63.38, "lon": -57, "region": "Antarctic Peninsula"}
        ],
        'images': [
            {
                "file": "Branchiostoma_lanceolatum.jpg",
                "species": "Branchiostoma lanceolatum",
                "vernacular": "lancelet",
                "description": "A Lancelet (or Amphioxus) specimen —Subphylum: Cephalochordata— collected in coarse sand sediments (600 µm) on the Belgian continental shelf. Total Length: approximately 22 mm. Geo-location not applicable as the picture was taken in the laboratory.",
                "attribution": "By © Hans Hillewaert, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=5712836",
                "location": None,
                "camera_coordinates": "",
                "date": "1997",
                "author": "Hans Hillewaert",
                "source": "",
                "sourcelink": "",
                "license": "CC4"
            },
            {
                "file": "Tunicate_komodo.jpg",
                "species": "Polycarpa aurata",
                "vernacular": "sea squirt",
                "description": "Komodo National Park sea squirt (Polycarpa aurata)",
                "attribution": "By Nhobgood Nick Hobgood - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=5633976",
                "location": None,
                "camera_coordinates": "",
                "date": "10 October 2006",
                "author": "Nhobgood Nick Hobgood",
                "source": "",
                "sourcelink": "https://commons.wikimedia.org/wiki/File:Tunicate_komodo.jpg",
                "license": "CC3"
            }
        ]
    },
    'Cnidaria': {
        'description': 'Aquatic invertebrates with stinging cells (cnidocytes) and radial symmetry; exist as polyps or medusae, often colonial.',
        'habitable_areas': 'Predominantly marine in shallow tropical waters, deep seas, and polar regions; common in coral reefs and open ocean.',
        'rarity': 'Common',  # >1M records
        'representative_species': 'Chrysaora fuscescens (Pacific sea nettle); Physalia physalis (Portuguese man o\' war)',
        'vernacular': 'Cnidarians',
        'gbif_taxon_id': '43',
        'hotspots': [
            {"lat": -16.5, "lon": 145.5, "region": "Great Barrier Reef"},
            {"lat": -0.5, "lon": -90.5, "region": "Galápagos (Coral/Anemone Spot)"},
            {"lat": 25.03, "lon": -78.04, "region": "Bahamas, Caribbean (Coral Reefs)"},
            {"lat": 3.25, "lon": 73, "region": "Maldives, Indian Ocean (Tropical Reefs)"},
            {"lat": 9.83, "lon": -104.3, "region": "East Pacific Rise (Hydrothermal Vents)"},
            {"lat": -75, "lon": -175, "region": "Ross Sea, Antarctic (Polar Seabeds)"}
        ],
        'images': [
            {
                "file": "west-coast-sea-nettle-2.jpg",
                "species": "Chrysaora fuscescens",
                "vernacular": "Pacific sea nettle (West Coast sea nettle)",
                "description": "Cultured at the Aquarium of the Pacific from polyp to adult ephyrae. The Aquarium habitat for west coast sea nettles is in the Northern Pacific Gallery. Our aquarists have successfully cultured this species for many years. It takes about three months to rear the jellies from polyps to ephyrae, the adult stage. We exhibit our Aquarium of the Pacific-grown jellies and also share them with other aquariums. Pacific sea nettles (also known as West Coast sea nettles) are in the class Scyphozoa, that of the jellies called true jellies. The genus name of sea nettle jellies, Chrysaora, comes from Greek mythology. Chrysaor, reportedly a giant, was the son of Poseidon and Medusa. His name translates as ‘golden falchion’. A falchion was a commonly used curved fighting sword that could cut through armor, a reference to the stinging ability of these jellies. The West Coast sea nettle’s species name, fuscescens, means dusky or dark referring to the dusky color of the nettle’s bell.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "August 22, 2007",
                "author": "Unknown",
                "source": "",
                "sourcelink": "https://www.aquariumofpacific.org/onlinelearningcenter/species/pacific_sea_nettle",
                "license": "CC4"
            },
            {
                "file": "800px-Portuguese_Man-O-War_(Physalia_physalis).jpg",
                "species": "Physalia physalis",
                "vernacular": "Portuguese man-of-war",
                "description": "Portuguese man-of-war (Physalia physalis)",
                "attribution": "By Image courtesy of Islands in the Sea 2002, NOAA/OER. - U.S. Department of Commerce, National Oceanic and Atmospheric Administration, Public Domain, https://commons.wikimedia.org/w/index.php?curid=185562",
                "location": None,
                "camera_coordinates": "",
                "date": "2002",
                "author": "Unknown",
                "source": "U.S. Department of Commerce, National Oceanic and Atmospheric Administration. Image courtesy of Islands in the Sea 2002, NOAA/OER.",
                "sourcelink": "",
                "license": "public domain"
            }
        ]
    },
    'Ctenophora': {
        'description': 'Marine invertebrates with comb-like cilia for swimming and sticky colloblasts for prey capture; gelatinous and predatory.',
        'habitable_areas': 'Sea waters worldwide, from polar to tropical, near coasts to deep ocean; mostly planktonic.',
        'rarity': 'Medium',  # ~10K-50K records
        'representative_species': 'Mnemiopsis leidyi (sea walnut); Pleurobrachia bachei (sea gooseberry)',
        'vernacular': 'Comb jellies',
        'gbif_taxon_id': '51',
        'hotspots': [
            {"lat": 44, "lon": 35, "region": "Black Sea"},
            {"lat": 46, "lon": 35, "region": "Sea of Azov"},
            {"lat": 35, "lon": 30, "region": "Eastern Mediterranean"},
            {"lat": 55, "lon": 3, "region": "North Sea"},
            {"lat": 57, "lon": 20, "region": "Baltic Sea"},
            {"lat": 78, "lon": 16, "region": "Svalbard, Arctic"},
            {"lat": -75, "lon": -175, "region": "Ross Sea, Antarctic"}
        ],
        'images': [
            {
                "file": "Sea_walnut,_Boston_Aquarium_(cropped).jpg",
                "species": "Mnemiopsis leidyi",
                "vernacular": "Sea walnut",
                "description": "Sea walnut (Mnemiopsis leidyi) at the New England Aquarium, Boston MA",
                "attribution": "By Steven G. Johnson - commons, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=77199719",
                "location": None,
                "camera_coordinates": "",
                "date": "16 August 2008",
                "author": "Steven G. Johnson",
                "source": "",
                "sourcelink": "https://en.wikipedia.org/wiki/Mnemiopsis#/media/File:Sea_walnut,_Boston_Aquarium_(cropped).jpg",
                "license": "CC3"
            },
            {
                "file": "1024px-Blankenberge_Pleurobrachia_pileus.jpeg",
                "species": "Pleurobrachia bachei",
                "vernacular": "sea gooseberry",
                "description": "Pleurobrachia pileus, sea gooseberry on Blankenberge beach, Belgium",
                "attribution": "By Zeisterre - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=18795854",
                "location": "Blankenberge beach, Belgium",
                "camera_coordinates": "",
                "date": "22 March 2012",
                "author": "Zeisterre on Wikmedia.org",
                "source": "",
                "sourcelink": "https://commons.wikimedia.org/wiki/File:Blankenberge_Pleurobrachia_pileus.JPG",
                "license": "CC3"
            }
        ]
    },
    'Cycliophora': {
        'description': 'Commensal aquatic animals with sac-like bodies and complex life cycles; highly specialized and microscopic.',
        'habitable_areas': 'Marine, commensal on mouthparts of cold-water lobsters in North Atlantic and Mediterranean.',
        'rarity': 'Rare', # <100 records (only 2-3 species)
        'representative_species': 'Symbion pandora (no common name); Symbion americanus (no common name)',
        'vernacular': 'No common name (genus Symbion)',
        'gbif_taxon_id': '45',
        'hotspots': [
            {"lat": 50, "lon": -30, "lon": "North Atlantic"},
            {"lat": 40, "lon": 15, "region": "Mediterranean"}
        ],
        'images': [
            {
                "file": "CYC-000075_hab_Symbion_pandora_Paratype.tif.jpg",
                "species": "Symbion pandora",
                "vernacular": "",
                "description": "This closeup of a pandora’s mouth shows it is surrounded by tiny hairs called cilia. The animal eats by twirling these cilia, which pulls little bits of food into its mouth. A single blood cell from a fish or a crab can barely squeeze down a pandora’s throat.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "2023",
                "author": "Funch & Kristensen, 1995, Natural History Museum of Denmark",
                "source": "Eibye-Jacobsen D, Pavesi L, Schiøtte T, Sørensen M V, Olesen J (2025). NHMD Invertebrate Zoology Collection. Natural History Museum of Denmark. Occurrence dataset https://doi.org/10.15468/wodhis accessed via GBIF.org on 2025-10-29. https://www.gbif.org/occurrence/2012908531",
                "sourcelink": "https://www.gbif.org/occurrence/2012908531",
                "license": "CC4"
            },
            {
                "file": "CYC-000248_hab_Symbion_americanus_Paratype.tif.jpg",
                "species": "Symbion americanus",
                "vernacular": "",
                "description": "CYC-000248 hab Symbion americanus Paratype",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "2023",
                "author": "Eibye-Jacobsen D, Pavesi L, Schiøtte T, Sørensen M V, Olesen J",
                "source": "",
                "sourcelink": "https://www.gbif.org/occurrence/2012908575",
                "license": "CC4"
            }
        ]
    },
    'Echinodermata': {
        'description': 'Marine animals with radial symmetry, water vascular system, and calcareous endoskeleton; include starfish, urchins, etc.',
        'habitable_areas': 'Exclusively marine, from intertidal to abyssal depths worldwide; common on seabeds and coral reefs.',
        'rarity': 'Common',  # >500K records
        'representative_species': 'Fromia indica (starfish); Actinopyga echinites (sea cucumber)',
        'vernacular': 'Echinoderms',
        'gbif_taxon_id': '50',
        'hotspots': [
            {"lat": -16.5, "lon": 145.5, "region": "Great Barrier Reef"},
            {"lat": 25.03, "lon": -78.04, "region": "Bahamas, Caribbean"},
            {"lat": -75, "lon": -175, "region": "Ross Sea, Antarctic (Deep Sea)"},
            {"lat": 28.03, "lon": -111.77, "region": "Gulf of California"},
            {"lat": 37.77, "lon": -122.43, "region": "San Francisco Bay"}
        ],
        'images': [
            {
                "file": "Starfish_montage.png",
                "species": "Fromia indica",
                "vernacular": "starfish",
                "description": "Montage of various starfish species, on a black background.",
                "attribution": "By Top left: Klaus Rassinger (Museum Wiesbaden)Top right: Katie AhlfeldBottom right: Jon Zander (Digon3)Bottom left: Espen Rekdal - File:0059 Blauer Seestern.jpg – CC BY SA 4.0File:Echinaster serpentarius (USNM E28192) 001.jpeg – CC 0File:Horned Starfish Macro.JPG – CC BY SA 3.0File:Hymenaster pellucidus 02 Espen Rekdal.jpg – CC BY 4.0, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=158982544",
                "location": None,
                "camera_coordinates": "",
                "date": "2 February 2025",
                "author": "Top left: Klaus Rassinger (Museum Wiesbaden) Top right: Katie Ahlfeld Bottom right: Jon Zander (Digon3) Bottom left: Espen Rekdal",
                "source": "",
                "sourcelink": "https://commons.wikimedia.org/wiki/File:Starfish_montage.png",
                "license": "CC4"
            },
            {
                "file": "Johnsons_MBAF2.jpg",
                "species": "Parastichopus johnsoni",
                "vernacular": "Johnsons Sea Cucumber",
                "description": "The habitat of Johnsons sea cumber is the sandy seafloor at depths as deep as 400 meters (1,300 feet). This species relies on particles of food falling from the surface as marine snow and prey it can find in the seafloor sediment. “Swimming” is limited to flexing its muscular body, inching along the seafloor on its tube feet.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Unknown",
                "source": "Aquarium of the Pacific. Credit: Used with permission of MBARI",
                "sourcelink": "https://www.aquariumofpacific.org/onlinelearningcenter/species/johnsons_sea_cucumber",
                "license": "CC4"
            }
        ]
    },
    'Entoprocta': {
        'description': 'Sessile aquatic animals with a goblet shape, crown of tentacles, and both mouth/anus inside the crown; mostly colonial.',
        'habitable_areas': 'Marine oceans on rocks, shells, algae; mostly shallow to 50m, some deep sea.',
        'rarity': 'Rare',  # <10K records
        'representative_species': 'Barentsia ramosa (no common name); Pedicellina cernua (no common name)',
        'vernacular': 'Kamptozoa',
        'gbif_taxon_id': '8173593',
        'hotspots': [
            {"lat": -63.38, "lon": -57, "region": "Antarctic Peninsula"},
            {"lat": 65.5, "lon": 38, "region": "White Sea"}
        ],
        'images': [
            {
                "file": "medium.jpeg",
                "species": "Barentsia ramosa",
                "vernacular": "",
                "description": "",
                "attribution": "",
                "location": "Carmel Pt., Monterey Co., CA",
                "camera_coordinates": "",
                "date": "November 2, 1975",
                "author": "Gary McDonald",
                "source": "",
                "sourcelink": "https://www.inaturalist.org/taxa/424380-Barentsia-ramosa",
                "license": "CC4"
            },
            {
                "file": "Figuur-1-Pedicellina-cernua-Foto-Marco-Faasse-Figure-1-Pedicellina-cernua-Photo.png",
                "species": "Pedicellina cernua",
                "vernacular": "",
                "description": "Faunistic overview of the entoprocts of the Netherlands (Entoprocta) Five species of entoprocts are known to occur in the Netherlands: Pedicellina cernua, Barentsia benedeni, Barentsia gracilis, B. matsushimana and B. ramosa. Barentsia ramosa is recorded here for the second time from Europe, from the Nieuwe Waterweg, leading to the port of Rotterda...",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Marco Faasse",
                "source": "",
                "sourcelink": "https://www.researchgate.net/figure/Figuur-1-Pedicellina-cernua-Foto-Marco-Faasse-Figure-1-Pedicellina-cernua-Photo_fig1_254892799",
                "license": "CC4"
            }
        ]
    },
    'Gastrotricha': {
        'description': 'Microscopic cylindrical animals with cilia, adhesive glands, and a muscular pharynx; hermaphrodites.',
        'habitable_areas': 'Marine sediments and interstitial spaces; benthic in sands and seabeds worldwide.',
        'rarity': 'Rare',  # <10K records
        'representative_species': 'Thaumastoderma ramuliferum (no common name); Lepidodermella squamatum (no common name)',
        'vernacular': 'Hairybellies or hairybacks',
        'gbif_taxon_id': '22',
        'hotspots': [
            {"lat": -33.92, "lon": 18.42, "region": "Cape Town, South Africa"},
            {"lat": 42.12, "lon": 15.5, "region": "Tremiti Archipelago, Adriatic"}
        ],
        'images': [
            {
                "file": "Thaumastoderma_ramuliferum.jpg",
                "species": "Thaumastoderma ramuliferum (Gastrotricha: Macridasyida)",
                "vernacular": "",
                "description": "SEM photomicrograph showing the general body shape and aspects of the cuticular covering of Thaumastoderma ramuliferum (Gastrotricha: Macridasyida). Habitus in ventral view. Scale bar: 20 µm.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "M. Antonio Todaro, Tobias Kånneby, Matteo Dal Zotto, Ulf Jondelius",
                "source": "",
                "sourcelink": "https://commons.wikimedia.org/wiki/File:Thaumastoderma_ramuliferum.jpg",
                "license": "CC2"
            },
            {
                "file": "Lepidodermella_squamatum.jpg",
                "species": "Lepidodermella squamatum",
                "vernacular": "",
                "description": "Microphotograph of Lepidodermella squamatum (Gastrotricha: Chaetonotida)",
                "attribution": "By Giuseppe Vago - Flickr.com, CC BY 2.0, https://commons.wikimedia.org/w/index.php?curid=15440726",
                "location": None,
                "camera_coordinates": "",
                "date": "22 August 2010",
                "author": "Giuseppe Vago",
                "source": "",
                "sourcelink": "https://commons.wikimedia.org/wiki/File:Lepidodermella_squamatum.jpg",
                "license": "CC2"
            }
        ]
    },
    'Gnathostomulida': {
        'description': 'Microscopic marine animals with cuticular jaws, no body cavity, and simultaneous hermaphroditism.',
        'habitable_areas': 'Shallow coastal sands and muds; anoxic-tolerant benthic environments.',
        'rarity': 'Rare',  # <1K records
        'representative_species': 'Gnathostomula paradoxa (no common name)',
        'vernacular': 'Jaw worms',
        'gbif_taxon_id': '77',
        'hotspots': [
            {"lat": 58.94, "lon": 20.13, "region": "Baltic Sea (Interstitial Jaw Worm Habitats)"},
            {"lat": 15.33, "lon": -76.16, "region": "Caribbean Sea (Shallow Coastal Gnathostomulid Spot)"}
        ],
        'images': [
            {
                "file": "fig028.jpg",
                "species": "Gnathostomula jenneri ",
                "vernacular": "",
                "description": "A tiny member of the interstitial fauna between grains of sand or mud. Species in this family are among the most commonly encountered jaw worms, found in shallow water and down to depths of several hundred meters.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "",
                "source": "",
                "sourcelink": "https://biocyclopedia.com/index/general_zoology/phylum_gnathostomulida.php",
                "license": "CC4"
            },
            {
                "file": "",
                "species": "",
                "vernacular": "",
                "description": "",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Unknown",
                "source": "",
                "sourcelink": "",
                "license": "CC BY 4.0"
            }
        ]
    },
    'Hemichordata': {
        'description': 'Marine deuterostomes with a proboscis, collar, and trunk; filter or deposit feeders.',
        'habitable_areas': 'Marine sediments and deep-sea; burrowing or colonial in oceans worldwide.',
        'rarity': 'Medium',  # ~10K records
        'representative_species': 'Saccoglossus kowalevskii (acorn worm); Cephalodiscus nigrescens (no common name)',
        'vernacular': 'Hemichordates (acorn worms, pterobranchs)',
        'gbif_taxon_id': '75',
        'hotspots': [
            {"lat": 50.37, "lon": -4.14, "region": "Plymouth, England"},
            {"lat": 32.3, "lon": -64.79, "region": "Bermuda"}
        ],
        'images': [
            {
                "file": "Enteropneusta.png",
                "species": "Saccoglossus kowalevskii",
                "vernacular": "acorn worm",
                "description": "1. Ptychodera clavigera, dorsal view. 1/l. After nature. 2. Ptychodera minuta q 1, from the ventral side. 2 /i- After life. 3. Ibid. Q, from the dorsal side. 2 /j. After life. 4. Ptyclwdera erythraea, from the dorsal side. Vi- Based on the existing fragments of the only specimen preserved in alcohol. 5. ScMzocardium brasiliense, from the side. After life. 6. Glandiceps talaboti. Forebody, from the dorsal side, '/i* After life. 7. Glandiceps haeksi, adult, forebody from the ventral side. Vi- After the specimen preserved in alcohol available to me. 8. The same specimen from the dorsal side. 9. Young individual of the same species. 2 /i- Based on a specimen preserved in alcohol. 10. Balanoglossus Jcoicalevskii. 4 /i- Based on a live specimen. 11. Balanioglossus hupfferi. i / i . Based on a live specimen.",
                "attribution": "By Spengel, Johann Wilhelm - Spengel, Johann Wilhelm (1893) Die Enteropneusten des Golfes von Neapel und der angrenzenden Meeres-Abschnitte Berlin : Verlag von R. Friedländer & Sohn, Public Domain, https://commons.wikimedia.org/w/index.php?curid=18957647",
                "location": None,
                "camera_coordinates": "",
                "date": "1893",
                "author": "Spengel, Johann Wilhelm. The originals for Figs. 2, 3, 4, 7, and 8 were provided by Mr. O. Peters in Göttingen, for Fig. 5 by Prof. E. Selenka in Erlangen, for Fig. 6 by Mr. Merculiako in Naples, for Fig. 10 by Prof. Charles S. Minot in Boston, and for Figs. 1, 9, and 11 by myself.",
                "source": "Spengel, Johann Wilhelm (1893) Die Enteropneusten des Golfes von Neapel und der angrenzenden Meeres-Abschnitte Berlin : Verlag von R. Friedländer & Sohn",
                "sourcelink": "https://commons.wikimedia.org/wiki/File:Enteropneusta.png",
                "license": "public domain"
            },
            {
                "file": "Cephalodiscus_nigrescens.jpg",
                "species": "Cephalodiscus nigrescens",
                "vernacular": "",
                "description": "The pterobranch Cephalodiscus nigrescens (wet specimen preserved in ethanol). Specimen collected by J. Tyler on 28 Jan 1959 with a bottom trawl in the (specifically Coats Land, Vahsel Bay, Off Duke Ernst Bay, at a depth of 393 m)",
                "attribution": "",
                "location": None,
                "camera_coordinates": "77° 40′ 12″ S, 35° 30′ 00″ W",
                "date": "31 March 2009",
                "author": "Adrian James Testa",
                "source": "https://commons.wikimedia.org/wiki/File:Cephalodiscus_nigrescens.jpg",
                "sourcelink": "https://collections.nmnh.si.edu/search/iz/?ark=ark:/65665/3db6c2545c10b444189b382fd11ec9e96",
                "license": "CC0"
            }
        ]
    },
    'Kinorhyncha': {
        'description': 'Small segmented marine invertebrates with a spiny introvert for locomotion; meiobenthic.',
        'habitable_areas': 'Marine mud and sand at all depths worldwide.',
        'rarity': 'Rare',  # <10K records
        'representative_species': 'Echinoderes hwiizaa (no common name); Echinoderes spinifurca (no common name)',
        'vernacular': 'Mud dragons',
        'gbif_taxon_id': '5959089',
        'hotspots': [
            {"lat": 7.34, "lon": -128.69, "region": "Clarion-Clipperton Fracture Zone (Abyssal Mud Dragon Habitats)"},
            {"lat": 11.2, "lon": 95.66, "region": "Andaman Sea (Intertidal Kinorhynch Diversity)"}
        ],
        'images': [
            {
                "file": "Echinoderes_hwiizaa.jpg",
                "species": "Echinoderes hwiizaa",
                "vernacular": "",
                "description": "Echinoderes coulli group (Echinoderidae, Cyclorhagida, Kinorhyncha) from the Ryukyu Islands, Japan. ZooKeys 382: 27–52.",
                "attribution": "By Hiroshi Yamasaki & Shinta Fujimoto - Two new species in the Echinoderes coulli group (Echinoderidae, Cyclorhagida, Kinorhyncha) from the Ryukyu Islands, Japan. ZooKeys 382: 27–52., CC BY 3.0, https://commons.wikimedia.org/w/index.php?curid=31265591",
                "location": "Ryukyu Islands, Japan",
                "camera_coordinates": "",
                "date": "20 February 2014",
                "author": "Hiroshi Yamasaki & Shinta Fujimoto",
                "source": "https://commons.wikimedia.org/wiki/File:Echinoderes_hwiizaa.jpg",
                "sourcelink": "",
                "license": "CC3"
            },
            {
                "file": "Echinoderes_spinifurca.png",
                "species": "Echinoderes spinifurca",
                "vernacular": "",
                "description": "",
                "attribution": "By Herranz M, Boyle M, Pardos F, Neves R - This file was derived from: Comparative-myoanatomy-of-Echinoderes-(Kinorhyncha)-a-comprehensive-investigation-by-CLSM-and-3D-1742-9994-11-31-S2.ogv, CC BY 2.0, https://commons.wikimedia.org/w/index.php?curid=148267740",
                "location": None,
                "camera_coordinates": "",
                "date": "2014",
                "author": "Herranz M, Boyle M, Pardos F, Neves R",
                "source": "This file was derived from: Comparative-myoanatomy-of-Echinoderes-(Kinorhyncha)-a-comprehensive-investigation-by-CLSM-and-3D-1742-9994-11-31-S2.ogv",
                "sourcelink": "",
                "license": "CC2"
            }
        ]
    },
    'Loricifera': {
        'description': 'Microscopic sediment-dwellers with a protective lorica and complex life cycles; anoxia-tolerant.',
        'habitable_areas': 'Marine sediments from shallow to deep sea; often in anoxic basins.',
        'rarity': 'Rare',  # <1K records (29 species)
        'representative_species': 'Pliciloricus enigmaticus (no common name); Spinoloricus cinziae (no common name)',
        'vernacular': 'No common name',
        'gbif_taxon_id': '5967457',
        'hotspots': [
            {"lat": 35.18, "lon": 21.41, "region": "L'Atalante Basin, Mediterranean"},
            {"lat": 30, "lon": -28.5, "region": "Great Meteor Seamount, Atlantic"},
            {"lat": 48.72, "lon": -3.99, "region": "Roscoff, France"},
            {"lat": 61.89, "lon": -6.91, "region": "Faroe Bank, North Atlantic"}
        ],
        'images': [
            {
                "file": "Pliciloricus_enigmatus.jpg",
                "species": "Pliciloricus enigmatus",
                "vernacular": "",
                "description": "Illustration by Carolyn Gast, National Museum of Natural History. From a condensed Smithsonian report, New Loricifera from Southeastern United States Coastal Waters.",
                "attribution": "By Carolyn Gast, National Museum of Natural History - http://seawifs.gsfc.nasa.gov/OCEAN_PLANET/HTML/oceanography_recently_revealed2.html, Public Domain, https://commons.wikimedia.org/w/index.php?curid=530608",
                "location": None,
                "camera_coordinates": "",
                "date": "26 August 2004",
                "author": "Carolyn Gast, National Museum of Natural History",
                "source": "https://commons.wikimedia.org/wiki/File:Pliciloricus_enigmatus.jpg",
                "sourcelink": "https://seawifs.gsfc.nasa.gov/OCEAN_PLANET/HTML/oceanography_recently_revealed2.html",
                "license": "public domain"
            },
            {
                "file": "Spinoloricus.png",
                "species": "Spinoloricus",
                "vernacular": "",
                "description": "Light microscopy image of the undescribed species of Spinoloricus (Loricifera; stained with Rose Bengal). Scale bar is 50 μm.",
                "attribution": "By Roberto Danovaro, Antonio Dell'Anno, Antonio Pusceddu, Cristina Gambi, Iben Heiner & Reinhardt Mobjerg Kristensen - Danovaro R., Dell'Anno A., Pusceddu A., Gambi C., Heiner I. & Kristensen R. M. (2010). 'The first metazoa living in permanently anoxic conditions'. BMC Biology 8: 30. doi:10.1186/1741-7007-8-30. Imported in 300dpi from http://www.biomedcentral.com/content/pdf/1741-7007-8-30.pdf Figure 1c, retouched., CC BY 2.0, https://commons.wikimedia.org/w/index.php?curid=9955335",
                "location": None,
                "camera_coordinates": "",
                "date": "6 April 2010",
                "author": "Roberto Danovaro, Antonio Dell'Anno, Antonio Pusceddu, Cristina Gambi, Iben Heiner & Reinhardt Mobjerg Kristensen",
                "source": "https://commons.wikimedia.org/wiki/File:Spinoloricus.png",
                "sourcelink": "Danovaro R., Dell'Anno A., Pusceddu A., Gambi C., Heiner I. & Kristensen R. M. (2010). 'The first metazoa living in permanently anoxic conditions'. BMC Biology 8: 30. doi:10.1186/1741-7007-8-30. Imported in 300dpi from http://www.biomedcentral.com/content/pdf/1741-7007-8-30.pdf Figure 1c, retouched.",
                "license": "CC2"
            }
        ]
    },
    'Mollusca': {
        'description': 'Soft-bodied invertebrates with a mantle, radula, and often a shell; diverse classes like gastropods, bivalves.',
        'habitable_areas': 'Largest marine phylum; oceans worldwide from shores to abyssal zones.',
        'rarity': 'Common',  # >3M records
        'representative_species': 'Nautilus pompilius (Nautilus pompilius); Ruditapes philippinarum (Manila clam)',
        'vernacular': 'Mollusks',
        'gbif_taxon_id': '52',
        'hotspots': [
            {"lat": 28.03, "lon": -111.77, "region": "Gulf of California"},
            {"lat": -16.5, "lon": 145.5, "region": "Great Barrier Reef"},
            {"lat": -34.93, "lon": 138.6, "region": "Adelaide, South Australia"},
            {"lat": 37.77, "lon": -122.43, "region": "San Francisco Bay"},
            {"lat": 21.31, "lon": -157.86, "region": "Honolulu, Hawaii (Pacific)"},
            {"lat": 49.25, "lon": -123.12, "region": "Vancouver Coast"}
        ],
        'images': [
            {
                "file": "Nautilus_belauensis_from_Palau.jpg",
                "species": "Nautilus pompilius",
                "vernacular": "Nautilus pompilius",
                "description": "Nautilus, Palau",
                "attribution": "By Manuae - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=18395466",
                "location": "Nautilus, Palau",
                "camera_coordinates": "",
                "date": "7 February 2012",
                "author": "Manuae on Wikimedia",
                "source": "https://commons.wikimedia.org/wiki/File:Nautilus_belauensis_from_Palau.jpg",
                "sourcelink": "",
                "license": "CC3"
            },
            {
                "file": "Ruditapes_philippinarum.jpg",
                "species": "Ruditapes philippinarum",
                "vernacular": "Manila clam",
                "description": "Ruditapes philippinarum (or Tapes (Ruditapes) philippinarum or Tapes philippinarum or Venerupis (Ruditapes) philippinarum) (A.Adams et Reeve, 1850) (Bivalvia: Heterodonta: Veneridae) harvested in Ise Bay, Mie Prefecture, Honshū Island, Japan.",
                "attribution": "By Original file, Public Domain, https://commons.wikimedia.org/w/index.php?curid=1382703",
                "location": "Honshū Island, Japan",
                "camera_coordinates": "",
                "date": "",
                "author": "Unknown",
                "source": "https://commons.wikimedia.org/wiki/File:Ruditapes_philippinarum.jpg",
                "sourcelink": "",
                "license": "public domain"
            }
        ]
    },
    'Nematoda': {
        'description': 'Slender unsegmented worms with a cuticle and tubular gut; free-living or parasitic.',
        'habitable_areas': 'Abundant in marine sediments and ocean floors; meiobenthic worldwide.',
        'rarity': 'Medium',  # ~200K records
        'representative_species': 'Caenorhabditis elegans (model roundworm); Ascaris lumbricoides (human roundworm)',
        'vernacular': 'Roundworms',
        'gbif_taxon_id': '5967481',
        'hotspots': [
            {"lat": 55, "lon": 3, "region": "North Sea (Ocean Floor)"},
            {"lat": 57, "lon": 20, "region": "Baltic Sea"},
            {"lat": 44, "lon": 35, "region": "Black Sea"}
        ],
        'images': [
            {
                "file": "merlin_157432140_f91af05d-c96a-454b-8d2c-c0d75e238478-jumbo.webp",
                "species": "Caenorhabditis elegans",
                "vernacular": "roundworm",
                "description": "a free-living transparent nematode about 1 mm in length that lives in temperate soil environments.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Unknown",
                "source": "Gschmeissner/Science Source",
                "sourcelink": "",
                "license": "CC4"
            },
            {
                "file": "shutterstock_451845385_1.webp",
                "species": "Helminthiasis Toxocara canis",
                "vernacular": "dog roundworm",
                "description": "The life cycle of the roundworm begins when eggs are passed with feces and deposited in the soil. It takes two to four weeks for the eggs to become infectious. A new host is infected by ingesting the eggs. The eggs hatch, releasing larvae that penetrate the walls of the small intestine and enter the bloodstream. The larvae can travel to organs throughout the body from the blood. In its natural host, the roundworm completes its life cycle by returning to the small intestine and developing into adult worms. The process takes about 60 to 90 days after hatching. The worms then mate and produce eggs that are excreted in the feces, beginning the cycle anew. The incubation period can be extended in lower temperatures, and in northern climates, the eggs may remain dormant through the winter.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Unknown",
                "source": "Copyright MRAORAOR/Shutterstock",
                "sourcelink": "https://www.news-medical.net/health/Roundworm-Transmission-From-Pets-To-Humans.aspx",
                "license": "CC4"
            }
        ]
    },
    'Nematomorpha': {
        'description': 'Parasitoid worms similar to nematodes; larvae parasitic on arthropods, adults free-living.',
        'habitable_areas': 'Marine (planktonic adults, parasitic larvae in crustaceans); some freshwater.',
        'rarity': 'Rare',  # <1K records
        'representative_species': 'Paragordius tricuspidatus (no common name); Nectonema sp. (no common name)',
        'vernacular': 'Horsehair worms or Gordian worms',
        'gbif_taxon_id': '64',
        'hotspots': [
            {"lat": 34.55, "lon": 18.05, "region": "Mediterranean Sea (Marine Horsehair Worm Habitats)"},
            {"lat": 31.78, "lon": -40.25, "region": "North Atlantic Ocean (Pelagic Nematomorpha Spot)"}
        ],
        'images': [
            {
                "file": "Paragordius_tricuspidatus.jpeg",
                "species": "Paragordius tricuspidatus",
                "vernacular": "horsehair worm",
                "description": "A horsehair worm (Phylum Nematomorpha), species Paragordius tricuspidatus. Hairworms are water parasites that invade crickets.",
                "attribution": "By Bildspende von D. Andreas Schmidt-Rhaesa, Veröffentlichung unter GNU FDL -- Necrophorus 15:31, 8. Sep 2004 (CEST) - Transferred from de.wikipedia to Commons., Bild:Paragordius_tricuspidatus.jpeg.jpg, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=391161",
                "location": None,
                "camera_coordinates": "",
                "date": "8 September 2004",
                "author": "Bildspende von D. Andreas Schmidt-Rhaesa, Veröffentlichung unter GNU FDL -- Necrophorus 15:31, 8. Sep 2004 (CEST)",
                "source": "https://commons.wikimedia.org/wiki/File:Paragordius_tricuspidatus.jpeg",
                "sourcelink": "",
                "license": "GNU"
            },
            {
                "file": "Nectonema-sp-parasitic-on-Natatolana-japonensis-a-bN-japonensis-containing-Nectonema.png",
                "species": "Nectonema sp.",
                "vernacular": "",
                "description": "Nectonema sp. parasitic on Natatolana japonensis. a, bN. japonensis containing Nectonema sp., dorsal and ventral views, fresh specimen (SMBL-V0598). cNectonema sp., fresh specimen (ICHUM-6178)",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "Jul 2021",
                "author": "Keiichi Kakui, Jun Fukuchi, Daisuke Shimada",
                "source": "",
                "sourcelink": "https://www.researchgate.net/figure/Nectonema-sp-parasitic-on-Natatolana-japonensis-a-bN-japonensis-containing-Nectonema_fig1_352666099",
                "license": "researchgate"
            }
        ]
    },
    'Nemertea': {
        'description': 'Unsegmented worms with an eversible venomous proboscis; carnivorous.',
        'habitable_areas': 'Mostly marine in sediments, crevices, and open ocean; some pelagic.',
        'rarity': 'Medium',  # ~50K records
        'representative_species': 'Lineus longissimus (bootlace worm); Carcinonemertes errans (no common name)',
        'vernacular': 'Ribbon worms or proboscis worms',
        'gbif_taxon_id': '63',
        'hotspots': [
            {"lat": 37.77, "lon": -122.43, "region": "San Francisco Coastal"},
            {"lat": 21.31, "lon": -157.86, "region": "Honolulu, Hawaii"},
            {"lat": -33.92, "lon": 18.42, "region": "Cape Town, South Africa"},
            {"lat": 55, "lon": 3, "region": "North Sea"},
            {"lat": 28.03, "lon": -111.77, "region": "Gulf of California"}
        ],
        'images': [
            {
                "file": "Lineus_longissimus_retouched.jpg",
                "species": "Lineus longissimus",
                "vernacular": "bootlace worm",
                "description": "",
                "attribution": "By © Citron, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=81216433",
                "location": None,
                "camera_coordinates": "",
                "date": "29 July 2010",
                "author": "Citron, retouched by Habitator terrae",
                "source": "https://commons.wikimedia.org/wiki/File:Lineus_longissimus_retouched.jpg",
                "sourcelink": "",
                "license": "CC3"
            },
            {
                "file": "image-asset.webp",
                "species": "Carcinonemertes errans",
                "vernacular": "",
                "description": "Carcinonemertes errans larva collected in January 2013, note four eyes (left). Juvenile C. errans which was collected from an adult male Dungeness crab. Note two eyes (right). Scale bars 100 µm.",
                "attribution": "",
                "location": "Coos Bay: Jan-Mar, July, Sep-Oct, Dec",
                "camera_coordinates": "",
                "date": "",
                "author": "Unknown",
                "source": "",
                "sourcelink": "https://www.nemerteanlarvalid.com/carcinonemerteserrans",
                "license": ""
            }
        ]
    },
    'Orthonectida': {
        'description': 'Simple multicellular parasites with ciliated cells; wormlike and microscopic.',
        'habitable_areas': 'Marine, parasitic in invertebrates like flatworms and mollusks.',
        'rarity': 'Rare',  # <100 records
        'representative_species': 'Rhopalura ophiocomae (no common name); Intoshia linei (no common name)',
        'vernacular': 'Orthonectids',
        'gbif_taxon_id': '5967456',
        'hotspots': [
            {"lat": 65.65, "lon": 36.85, "region": "White Sea (Parasitic Orthonectid Habitats)"},
            {"lat": 47.6, "lon": -122.45, "region": "Puget Sound (Invertebrate Host Diversity)"}
        ],
        'images': [
            {
                "file": "Rhopalura.jpg",
                "species": "Rhopalura sp.",
                "vernacular": "",
                "description": "Rhopalura sp., above, is a parasite of Platyhelminthes, Mollusca and Annelida. Its body is simply a layer of epithelial cells, many with cilia, surrounding a group of sex cells.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Unknown",
                "source": "",
                "sourcelink": "https://www.bumblebee.org/invertebrates/MESOZOA.htm",
                "license": "CC4"
            },
            {
                "file": "jmor21602-fig-0002-m.jpg",
                "species": "Intoshia linei",
                "vernacular": "",
                "description": "Orthonectids are enigmatic parasitic bilaterians whose exact position on the phylogenetic tree is still uncertain.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "2023",
                "author": "Elizaveta K. Skalon, Viktor V. Starunov, Natalya I. Bondarenko, George S. Slyusarev",
                "source": "Skalon, E. K., Starunov, V. V., Bondarenko, N. I., & Slyusarev, G. S. (2023). Plasmodium structure of Intoshia linei (Orthonectida). Journal of Morphology, 284, e21602. https://doi.org/10.1002/jmor.21602",
                "sourcelink": "https://onlinelibrary.wiley.com/doi/abs/10.1002/jmor.21602",
                "license": ""
            }
        ]
    },
    'Phoronida': {
        'description': 'Tube-dwelling filter-feeders with a lophophore; U-shaped gut.',
        'habitable_areas': 'Marine sediments, rocks, and shells; intertidal to 400m depth worldwide.',
        'rarity': 'Medium',  # ~10K records
        'representative_species': 'Phoronis sp. (horseshoe worm); Phoronopsis harmeri (no common name)',
        'vernacular': 'Horseshoe worms',
        'gbif_taxon_id': '19',
        'hotspots': [
            {"lat": 65.5, "lon": 38, "region": "White Sea, Arctic"},
            {"lat": -23.65, "lon": -70.4, "region": "California Coastal"},
            {"lat": 43, "lon": 131, "region": "Sea of Japan"},
            {"lat": 22.63, "lon": 120.27, "region": "South China Sea"},
            {"lat": 21.31, "lon": -157.86, "region": "Hawaii"}
        ],
        'images': [
            {
                "file": "Phoronis_hippocrepia_2_Wright,_1856.jpg",
                "species": "Phoronis hippocrepia",
                "vernacular": "horseshoe worm",
                "description": "Phoronis hippocrepia Wright, 1856 - Banyuls-sur-Mer : 07/1990",
                "attribution": "By Géry PARENT - Own work, Public Domain, https://commons.wikimedia.org/w/index.php?curid=10547042",
                "location": None,
                "camera_coordinates": "",
                "date": "1990",
                "author": "Géry PARENT ",
                "source": "https://commons.wikimedia.org/wiki/File:Phoronis_hippocrepia_2_Wright,_1856.jpg",
                "sourcelink": "",
                "license": "public domain"
            },
            {
                "file": "Phoronopsis_harmeri_IZ_1643662.png",
                "species": "Phoronopsis harmeri",
                "vernacular": "",
                "description": "Phoronopsis harmeri IZ 1643662",
                "attribution": "By Gustav Paulay - https://www.gbif.org/occurrence/3117128401, CC0, https://commons.wikimedia.org/w/index.php?curid=131663107",
                "location": None,
                "camera_coordinates": "",
                "date": "17 April 2019",
                "author": "Gustav Paulay",
                "source": "https://commons.wikimedia.org/wiki/File:Phoronopsis_harmeri_IZ_1643662.png",
                "sourcelink": "https://www.gbif.org/occurrence/3117128401",
                "license": "CC0"
            }
        ]
    },
    'Placozoa': {
        'description': 'Simple blob-like cell aggregations; no tissues or organs, feed by engulfment.',
        'habitable_areas': 'Marine seafloors and benthic zones globally.',
        'rarity': 'Rare',  # <100 records
        'representative_species': 'Trichoplax adhaerens (no common name); Hoilungia hongkongensis (no common name)',
        'vernacular': 'Flat animals',
        'gbif_taxon_id': '76',
        'hotspots': [
            {"lat": 35.1, "lon": 139.08, "region": "Seto Inland Sea, Japan"},
            {"lat": 22.36, "lon": 114.11, "region": "Hong Kong"},
            {"lat": 40.64, "lon": 14.38, "region": "Naples, Italy"},
            {"lat": 35.18, "lon": 21.41, "region": "Mediterranean"}
        ],
        'images': [
            {
                "file": "Trichoplax_adhaerens_photograph.png",
                "species": "Trichoplax adhaerens",
                "vernacular": "",
                "description": "The animal is about 0.5 mm in diameter",
                "attribution": "By Bernd Schierwater - Eitel M, Osigus H-J, DeSalle R, Schierwater B (2013) Global Diversity of the Placozoa. PLoS ONE 8(4): e57131. doi:10.1371/journal.pone.0057131, CC BY 4.0, https://commons.wikimedia.org/w/index.php?curid=35712628",
                "location": None,
                "camera_coordinates": "",
                "date": "2 April 2013",
                "author": "Bernd Schierwater",
                "source": "https://commons.wikimedia.org/wiki/File:Trichoplax_adhaerens_photograph.png",
                "sourcelink": "Eitel M, Osigus H-J, DeSalle R, Schierwater B (2013) Global Diversity of the Placozoa. PLoS ONE 8(4): e57131. doi:10.1371/journal.pone.0057131",
                "license": "CC4"
            },
            {
                "file": "Hoilungia_hongkongensis.jpg",
                "species": "Hoilungia hongkongensis",
                "vernacular": "",
                "description": "Scanning electron microscopic image of Hoilungia hongkongensis",
                "attribution": "By Tessler M, Neumann JS, Kamm K, Osigus HJ, Eshel G, Narechania A, Burns JA, DeSalle R, Schierwater B - https://doi.org/10.3389/fevo.2022.1016357, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=132540993",
                "location": None,
                "camera_coordinates": "",
                "date": "8 December 2022",
                "author": "Tessler M, Neumann JS, Kamm K, Osigus HJ, Eshel G, Narechania A, Burns JA, DeSalle R, Schierwater B",
                "source": "https://commons.wikimedia.org/wiki/File:Hoilungia_hongkongensis.jpg",
                "sourcelink": "https://doi.org/10.3389/fevo.2022.1016357",
                "license": "CC4"
            }
        ]
    },
    'Platyhelminthes': {
        'description': 'Soft-bodied acoelomates with one digestive opening; free-living or parasitic.',
        'habitable_areas': 'Marine waters as predators/scavengers; parasitic in fish/crustaceans.',
        'rarity': 'Medium',  # ~100K records
        'representative_species': 'Pseudobiceros hancockanus (sea slug flatworm); Fasciola hepatica (liver fluke)',
        'vernacular': 'Flatworms',
        'gbif_taxon_id': '108',
        'hotspots': [
            {"lat": 14, "lon": 120.97, "region": "Philippines"},
            {"lat": 3.25, "lon": 73, "region": "Maldives"},
            {"lat": -36.85, "lon": 174.76, "region": "New Zealand"},
            {"lat": 13.5, "lon": 144.8, "region": "Guam"},
            {"lat": -33.92, "lon": 18.42, "region": "Cape Town, South Africa"}
        ],
        'images': [
            {
                "file": "Pseudobiceros_hancockanus.jpg",
                "species": "Pseudobiceros hancockanus",
                "vernacular": "sea slug flatworm",
                "description": "Marine flatworm Pseudobiceros gloriosus. Lembeh straits, North Sulawesi, Indonesia.",
                "attribution": "By Jens Petersen - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=3842102",
                "location": "Lembeh straits, North Sulawesi, Indonesia",
                "camera_coordinates": "",
                "date": "24 December 2007",
                "author": "Jens Petersen https://commons.wikimedia.org/wiki/User:Jnpet",
                "source": "https://commons.wikimedia.org/wiki/File:Pseudobiceros_hancockanus.jpg",
                "sourcelink": "https://commons.wikimedia.org/wiki/User:Jnpet",
                "license": "CC3"
            },
            {
                "file": "Fasciola_hepatica.jpeg",
                "species": "Fasciola hepatica",
                "vernacular": "liver fluke",
                "description": "Adult of Fasciola hepatica.",
                "attribution": "By I, Flukeman, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=2245831",
                "location": None,
                "camera_coordinates": "",
                "date": "5.6.2005",
                "author": "Flukeman on Wikimedia",
                "source": "https://commons.wikimedia.org/wiki/File:Fasciola_hepatica.JPG",
                "sourcelink": "",
                "license": "CC3"
            }
        ]
    },
    'Porifera': {
        'description': 'Porous filter-feeders with no tissues/organs; skeletons of spicules or spongin.',
        'habitable_areas': 'Marine worldwide, from tidal zones to deep sea; on rocks and sediments.',
        'rarity': 'Common',  # >300K records
        'representative_species': 'Aplysina archeri (stove-pipe sponge); Euplectella aspergillum (Venus\'s flower basket)',
        'vernacular': 'Sponges',
        'gbif_taxon_id': '105',
        'hotspots': [
            {"lat": 36.8, "lon": -122.0, "region": "Monterey Bay (Sponge Habitats)"},
            {"lat": -16.5, "lon": 145.5, "region": "Great Barrier Reef"},
            {"lat": 25.03, "lon": -78.04, "region": "Bahamas, Caribbean"},
            {"lat": 37.77, "lon": -122.43, "region": "California Deep Waters"},
            {"lat": 35.18, "lon": 21.41, "region": "Mediterranean Caves"},
            {"lat": -75, "lon": -175, "region": "Antarctic Regions"},
            {"lat": 28.03, "lon": -111.77, "region": "Deep Pacific"},
            {"lat": 29.53, "lon": 35.01, "region": "Northern Red Sea"}
        ],
        'images': [
            {
                "file": "Aplysina_archeri_(Stove-pipe_Sponge-pink_variation).jpg",
                "species": "Aplysina archeri",
                "vernacular": "stove-pipe sponge",
                "description": "Stove-pipe Sponge-pink variation.",
                "attribution": "By Nhobgood (talk) Nick Hobgood - Own work, CC BY-SA 3.0, https://commons.wikimedia.org/w/index.php?curid=11448769",
                "location": None,
                "camera_coordinates": "",
                "date": "13 December 2009",
                "author": "Nick Hobgood; Nhobgood on Wikimedia",
                "source": "https://en.wikipedia.org/wiki/File:Aplysina_archeri_(Stove-pipe_Sponge-pink_variation).jpg",
                "sourcelink": "",
                "license": "CC3"
            },
            {
                "file": "Euplectella_aspergillum_Okeanos.jpg",
                "species": "Euplectella aspergillum",
                "vernacular": "Venus's flower basket",
                "description": "A spectacular group of Venus flower basket glass sponges (Euplectella aspergillum) glass sponges with a squat lobster in the middle.",
                "attribution": "By NOAA Okeanos Explorer Program, Gulf of Mexico 2012 Expedition - http://www.photolib.noaa.gov/htmls/expl7519.htm, Public Domain, https://commons.wikimedia.org/w/index.php?curid=45669584",
                "location": None,
                "camera_coordinates": "",
                "date": "21 March 2012",
                "author": "NOAA Okeanos Explorer Program, Gulf of Mexico 2012 Expedition",
                "source": "https://commons.wikimedia.org/wiki/File:Euplectella_aspergillum_Okeanos.jpg",
                "sourcelink": "http://www.photolib.noaa.gov/htmls/expl7519.htm",
                "license": "public domain"
            }
        ]
    },
    'Priapulida': {
        'description': 'Unsegmented marine worms with a spiny introvert; carnivorous or detritivorous.',
        'habitable_areas': 'Marine mud/sand from shallow to deep; cold waters up to 13°C.',
        'rarity': 'Rare',  # <10K records
        'representative_species': 'Priapulus caudatus (cactus worm); Halicryptus spinulosus (no common name)',
        'vernacular': 'Penis worms or priapulid worms',
        'gbif_taxon_id': '5963150',
        'hotspots': [
            {"lat": 60, "lon": -150, "region": "Alaskan Bay"},
            {"lat": -75, "lon": -175, "region": "Antarctic"},
            {"lat": -18.14, "lon": 178.44, "region": "Fiji"}
        ],
        'images': [
            {
                "file": "Priapulus_caudatus.jpg",
                "species": "",
                "vernacular": "",
                "description": "Priapulid worm Priapulus caudatus in a Petri dish. The specimen was found in the intertidal of the Russian coast of the Barents Sea.",
                "attribution": "By Shunkina Ksenia - kindly granted by the author, CC BY 3.0, https://commons.wikimedia.org/w/index.php?curid=8747085",
                "location": None,
                "camera_coordinates": "",
                "date": "20 September 2004",
                "author": "Shunkina Ksenia",
                "source": "https://commons.wikimedia.org/wiki/File:Priapulus_caudatus.jpg",
                "sourcelink": "",
                "license": "CC3"
            },
            {
                "file": "Halicryptus_spinulosus_1.jpeg",
                "species": "Halicryptus spinulosus",
                "vernacular": "priapulid worm",
                "description": "Adult with fully protruded introvert of the priapulid worm Halicryptus spinulosus",
                "attribution": "By Ralf Janssen, Sofia A Wennberg and Graham E Budd - http://www.frontiersinzoology.com/content/6/1/8/figure/F1, CC BY 2.5, https://commons.wikimedia.org/w/index.php?curid=12397220",
                "location": None,
                "camera_coordinates": "",
                "date": "26 May 2009",
                "author": "Ralf Janssen, Sofia A Wennberg and Graham E Budd",
                "source": "https://commons.wikimedia.org/wiki/File:Halicryptus_spinulosus_1.JPEG",
                "sourcelink": "http://www.frontiersinzoology.com/content/6/1/8/figure/F1",
                "license": "CC2"
            }
        ]
    },
    'Rhombozoa': {
        'description': 'Tiny parasites in cephalopod kidneys; simple structure with axial cell and ciliated jacket.',
        'habitable_areas': 'Marine, parasitic in cephalopod renal appendages; temperate benthic zones.',
        'rarity': 'Rare',  # <100 records
        'representative_species': 'Dicyema japonicum (no common name); Dicyema misakiense (no common name)',
        'vernacular': 'Rhombozoans (or dicyemids)',
        'gbif_taxon_id': '7663989',
        'hotspots': [
            {"lat": 35.1, "lon": 139.08, "region": "Japan (Temperate Benthic)"}
        ],
        'images': [
            {
                "file": "Dicyema_japonicum.png",
                "species": "Dicyema japonicum",
                "vernacular": "",
                "description": "This photo shows Dicyema japonicum Furuya and Tsuneki, 1992 from Octopus sinensis d'Orbigny, 1841 (Japanese common octopus) taken by Hidetaka Furuya, the researcher of Dicyemida.",
                "attribution": "By 古屋秀隆 (Hidetaka Furuya) - 私信にて、ウィキメディアに掲載するために提供して頂いた。I received it from Dr. Hidedaka Furuya for Wikimedia Commons., CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=82811532",
                "location": None,
                "camera_coordinates": "",
                "date": "3 October 2019",
                "author": "古屋秀隆 (Hidetaka Furuya)",
                "source": "https://commons.wikimedia.org/wiki/File:Dicyema_japonicum.png",
                "sourcelink": "I received it from Dr. Hidedaka Furuya for Wikimedia Commons.",
                "license": "CC4"
            },
            {
                "file": "Dicyema_macrocephalum.png",
                "species": "Dicyema macrocephalum",
                "vernacular": "",
                "description": "Morphlogy of Dicyema macrocephalum (nematogen stage).",
                "attribution": "By Энциклопедический словарь Брокгауза и Ефрона (Brockhaus and Efron Encyclopedic Dictionary).Reprint by the Russian State Library., Public Domain, https://commons.wikimedia.org/w/index.php?curid=16680182",
                "location": None,
                "camera_coordinates": "",
                "date": "1890—1907",
                "author": "Unknown",
                "source": "https://commons.wikimedia.org/wiki/File:Dicyema_macrocephalum.png",
                "sourcelink": "Энциклопедический словарь Брокгауза и Ефрона (Brockhaus and Efron Encyclopedic Dictionary). Reprint by the Russian State Library.",
                "license": "public domain"
            }
        ]
    },
    'Rotifera': {
        'description': 'Microscopic pseudocoelomates with a ciliated corona for feeding/locomotion.',
        'habitable_areas': 'Mostly freshwater, but some marine as zooplankton.',
        'rarity': 'Medium',  # ~50K records
        'representative_species': 'Brachionus plicatilis (no common name); Bdelloid rotifer (no common name)',
        'vernacular': 'Wheel animals',
        'gbif_taxon_id': '91',
        'hotspots': [
            {"lat": 44.0, "lon": -124.1, "region": "Oregon Coast (Marine Rotifer Habitats)"},
            {"lat": -18.16, "lon": 147.49, "region": "Great Barrier Reef (Planktonic Rotifer Diversity)"}
        ],
        'images': [
            {
                "file": "Brachionus_plicatilis.jpg",
                "species": "Brachionus plicatilis",
                "vernacular": "",
                "description": "Rotifer Brachionus plicatilis",
                "attribution": "By Sofdrakou - Own work, CC BY-SA 4.0, https://commons.wikimedia.org/w/index.php?curid=45006975",
                "location": None,
                "camera_coordinates": "",
                "date": "16 September 2015",
                "author": "Sofdrakou on Wikimedia",
                "source": "",
                "sourcelink": "",
                "license": "CC4"
            },
            {
                "file": "Mikrofoto.de-Raedertier-14.jpg",
                "species": "Rotifera",
                "vernacular": "wheel animal",
                "description": "A Rotifera (wheel animal)",
                "attribution": "By Frank Fox - http://www.mikro-foto.de, CC BY-SA 3.0 de, https://commons.wikimedia.org/w/index.php?curid=20228899",
                "location": None,
                "camera_coordinates": "",
                "date": "8 April 2011",
                "author": "Frank Fox",
                "source": "https://commons.wikimedia.org/wiki/File:Mikrofoto.de-Raedertier-14.jpg",
                "sourcelink": "http://www.mikro-foto.de/",
                "license": "CC3"
            }
        ]
    },
    'Sipuncula': {
        'description': 'Unsegmented annelids with a retractable introvert; deposit feeders.',
        'habitable_areas': 'Marine benthic worldwide; burrows in sand/mud, under stones, to abyssal depths.',
        'rarity': 'Medium',  # ~10K records
        'representative_species': 'Thysanocardia nigra (no common name); Sipunculus nudus (peanut worm)',
        'vernacular': 'Peanut worms or sipunculid worms',
        'gbif_taxon_id': '74',
        'hotspots': [
            {"lat": 21.31, "lon": -157.86, "region": "Hawaii"},
            {"lat": -33.92, "lon": 18.42, "region": "Cape Town, South Africa"},
            {"lat": 37.77, "lon": -122.43, "region": "San Francisco"}
        ],
        'images': [
            {
                "file": "Thysanocardia_nigra.jpg",
                "species": "Thysanocardia nigra",
                "vernacular": "",
                "description": "Thysanocardia nigra (Ikeda 1904)",
                "attribution": "By EcologyWA - https://www.flickr.com/photos/ecologywa/30470219132/, CC0, https://commons.wikimedia.org/w/index.php?curid=54913769",
                "location": None,
                "camera_coordinates": "",
                "date": "26 October 2016",
                "author": "EcologyWA",
                "source": "https://commons.wikimedia.org/wiki/File:Thysanocardia_nigra.jpg",
                "sourcelink": "https://www.flickr.com/photos/ecologywa/30470219132/",
                "license": "CC0"
            },
            {
                "file": "070421cyrg8928m3a.jpg",
                "species": "Sipunculus nudus",
                "vernacular": "peanut worm",
                "description": "Peanut worms are burrowing worm-like creatures that are sometimes seen above the ground on all our shores. When contracted, their ridged skins looks like the texture of peanut shells. Most are only a few millimeters long. Some burrow in mud, while others hide in crevices or abandoned snail shells and even in tubeworm tubes. 'Siphunculus' means 'little tube'. What is unique to peanut worms is their introvert, a long tube on their front end.",
                "attribution": "",
                "location": "Pulau Senang",
                "camera_coordinates": "",
                "date": "June 21, 2010",
                "author": "Loh Kok Sheng",
                "source": "https://www.flickr.com/photos/koksheng/4718959441/",
                "sourcelink": "http://www.wildsingapore.com/wildfacts/worm/sipuncula/sipuncula.htm",
                "license": "All Rights Reserved"
            }
        ]
    },
    'Tardigrada': {
        'description': 'Eight-legged micro-animals with extreme resilience via cryptobiosis.',
        'habitable_areas': 'Marine benthic, in sediments and on seaweeds; some deep-sea.',
        'rarity': 'Medium',  # ~10K records
        'representative_species': 'Milnesium tardigradum (no common name); Halobiotus crispae (no common name)',
        'vernacular': 'Water bears or moss piglets',
        'gbif_taxon_id': '14',
        'hotspots': [
            {"lat": -75, "lon": -175, "region": "Antarctic"},
            {"lat": 3.25, "lon": 73, "region": "Tropical Rainforests (Marine)"}
        ],
        'images': [
            {
                "file": "SEM_image_of_Milnesium_tardigradum_in_active_state_-_journal.pone.0045682.g001-2.png",
                "species": "",
                "vernacular": "",
                "description": "SEM image of Milnesium tardigradum in active state. doi:10.1371/journal.pone.0045682.g001",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "16 November 2012",
                "author": "Schokraie E, Warnken U, Hotz-Wagenblatt A, Grohme MA, Hengherr S, et al. (2012)",
                "source": "https://commons.wikimedia.org/wiki/File:SEM_image_of_Milnesium_tardigradum_in_active_state_-_journal.pone.0045682.g001-2.png",
                "sourcelink": "Schokraie E, Warnken U, Hotz-Wagenblatt A, Grohme MA, Hengherr S, et al. (2012) Comparative proteome analysis of Milnesium tardigradum in early embryonic state versus adults in active and anhydrobiotic state. PLoS ONE 7(9): e45682. doi:10.1371/journal.pone.0045682",
                "license": "CC2"
            },
            {
                "file": "SEM-investigation-of-Halobiotus-crispae-from-Vellerup-Vig-Denmark-A-Overview-of-P1.png",
                "species": "Halobiotus crispae",
                "vernacular": "",
                "description": "SEM investigation of Halobiotus crispae from Vellerup Vig, Denmark. (A) Overview of P1 stage indicating the areas shown in B and C. The thick outer cuticle functionally isolates the animal from the surroundings (scale bar=100 μm). (B) Close-up of the head region of P1. Notice that the mouth is closed by cuticular thickenings (scale bar=25 μm). (C) Close-up of the posterior area of the P1 stage. As shown for the mouth, the cloaca is closed (scale bar=10 μm). (D) Close-up of the head region of the active stage. Note the six peribuccal sensory organs (*) that surround the open mouth (scale bar=25 μm). (E) Close-up of the posterior area of the active stage, revealing the open tri-lobed cloaca (scale bar=10 μm).",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "",
                "author": "Kenneth Halberg",
                "source": "Cyclomorphosis in Tardigrada: Adaptation to environmental constraints: https://www.researchgate.net/publication/26745401_Cyclomorphosis_in_Tardigrada_Adaptation_to_environmental_constraints",
                "sourcelink": "https://www.researchgate.net/figure/SEM-investigation-of-Halobiotus-crispae-from-Vellerup-Vig-Denmark-A-Overview-of-P1_fig3_26745401",
                "license": "researchgate"
            }
        ]
    },
    'Xenoturbellida': {
        'description': 'Simple bilaterians with a ventral furrow and sac-like gut; no organs except statocyst.',
        'habitable_areas': 'Marine benthic, shallow to deep sea (up to 3700m).',
        'rarity': 'Rare',  # <100 records
        'representative_species': 'Xenoturbella bocki (no common name); Xenoturbella churro (no common name)',
        'vernacular': 'No common name',
        'gbif_taxon_id': '7190138',
        'hotspots': [
            {"lat": 58.33, "lon": 11.55, "region": "Gullmarsfjorden, Swedish West Coast"},
            {"lat": 35.1, "lon": 139.08, "region": "Western Pacific, Japan"}
        ],
        'images': [
            {
                "file": "Xenoturbella_bocki.jpg",
                "species": "Xenoturbella bocki",
                "vernacular": "",
                "description": "External morphology of Xenoturbella bocki. Side furrows (black arrow) are present on the lateral sides from the anterior tip (a), but these do not reach the posterior end (p). The mouth (black arrowhead) is situated on the ventral side anterior to the circumferential furrow (white arrowhead). Scale bar: 1 cm.",
                "attribution": "",
                "location": None,
                "camera_coordinates": "",
                "date": "2015",
                "author": "Hiroaki Nakano",
                "source": "https://commons.wikimedia.org/wiki/File:Xenoturbella_bocki.jpg",
                "sourcelink": "https://pmc.ncbi.nlm.nih.gov/articles/PMC4657256/figure/Fig1/",
                "license": "CC4"
            },
            {
                "file": "Xenoturbella_churro.jpg",
                "species": "Xenoturbella churro",
                "vernacular": "deep sea Giant Purple Sock Worm",
                "description": "Xenoturbella churro, also known as the deep sea Giant Purple Sock Worm.",
                "attribution": "By MBARI - MBARI, CC BY 4.0, https://commons.wikimedia.org/w/index.php?curid=171661421",
                "location": None,
                "camera_coordinates": "",
                "date": "February 3 2016",
                "author": "MBARI",
                "source": "https://commons.wikimedia.org/wiki/File:Xenoturbella_churro.jpg",
                "sourcelink": "",
                "license": "CC4"
            }
        ]
    }
}



# Display all phyla images in a grid if no phylum selected
st.subheader("Step 1: Select a Marine Phylum by Clicking 'Select'")
num_cols = 5 # Adjust number of columns for grid layout
selected_phylum = st.session_state.get('selected_phylum', None)
if not selected_phylum:
    for i in range(0, len(marine_phyla), num_cols):
        cols = st.columns(num_cols)
        for j, col in enumerate(cols):
            if i + j < len(marine_phyla):
                phylum = marine_phyla[i + j]
                images = phylum_info[phylum].get('images', [])
             
                if not images:
                    with col:
                        st.warning(f"No images found for {phylum}.")
                        st.write(f"{phylum} ({phylum_info[phylum]['vernacular']})")
                        with st.expander("Details"):
                            st.markdown(phylum_info[phylum]['description'])
                        if st.button("Select", key=f"select_{phylum}"):
                            st.session_state.selected_phylum = phylum
                            st.rerun()
                    continue
             
                # Initialize index in session state if not set
                idx_key = f"img_idx_{phylum}"
                if idx_key not in st.session_state:
                    st.session_state[idx_key] = 0
             
                current_idx = st.session_state[idx_key]
                current_img = images[current_idx]
             
                image_dir = "./data/phylumimgs/"
                image_path = os.path.join(image_dir, current_img['file'])
             
                with col:
                    if os.path.exists(image_path):
                        # Load and base64-encode the image for <img> tag
                        with open(image_path, "rb") as img_file:
                            img_data = img_file.read()
                            base64_img = base64.b64encode(img_data).decode('utf-8')
                            img_ext = os.path.splitext(image_path)[1][1:] # 'png' or 'jpg'
                     
                        description = current_img.get('description', '')
                        location = current_img.get('location', '')
                        author = current_img.get('author', '')
                        license = current_img.get('license', '')
                        source_link = current_img.get('source_link', '')
                        attribution = current_img.get('attribution', '') # Fallback full string
                     
                        # Build alt: description if available, else generic
                        alt_text = description if description else f"{phylum} Image {current_idx+1}"
                     
                        # Build title: combine all available fields
                        title_parts = []
                        if description:
                            title_parts.append(f"Description: {description}")
                        if location:
                            title_parts.append(f"Location: {location}")
                        if author:
                            title_parts.append(f"Author: {author}")
                        if license:
                            title_parts.append(f"License: {license}")
                        if source_link:
                            title_parts.append(f"Source: {source_link}")
                        if attribution and not (author or license or source_link): # Use full attribution if fields are missing
                            title_parts.append(attribution)
                        title_attr = f' title="{" | ".join(title_parts)}"' if title_parts else ''
                     
                        # Display image with alt and hover tooltip
                        st.markdown(
                            f'<img src="data:image/{img_ext};base64,{base64_img}" alt="{alt_text}"{title_attr} style="width:100%; height:auto;">',
                            unsafe_allow_html=True
                        )
                    else:
                        st.warning(f"Image not found: {current_img['file']}")
                 
                    # Vernacular below image
                    st.write(f"{phylum} ({phylum_info[phylum]['vernacular']})")
                 
                    # Cycling buttons if multiple images
                    if len(images) > 1:
                        btn_cols = st.columns(2)
                        with btn_cols[0]:
                            if st.button("Prev", key=f"prev_{phylum}_{i+j}"):
                                st.session_state[idx_key] = (current_idx - 1) % len(images)
                                st.rerun()
                        with btn_cols[1]:
                            if st.button("Next", key=f"next_{phylum}_{i+j}"):
                                st.session_state[idx_key] = (current_idx + 1) % len(images)
                                st.rerun()
                 
                    # Details expander (no images here; keep text-only)
                    with st.expander("Details"):
                        st.markdown(phylum_info[phylum]['description'])
                 
                    if st.button("Select", key=f"select_{phylum}"):
                        st.session_state.selected_phylum = phylum
                        selected_phylum = phylum
                        st.rerun() # Rerun to update the UI immediately
else:
    st.success(f"Selected: {selected_phylum} ({phylum_info[selected_phylum]['vernacular']})")
    with st.expander("Details"):
        st.markdown(phylum_info[selected_phylum]['description'])
     
        # Display all images for selected phylum
        images = phylum_info[selected_phylum].get('images', [])
        if images:
            st.subheader("Images")
            num_img_cols = min(3, len(images)) # Up to 3 columns for layout
            if len(images) > 1:
                img_cols = st.columns(num_img_cols)
            else:
                img_cols = [st] # Just use the main container if one
         
            for idx, img_data in enumerate(images):
                image_path = os.path.join("./data/phylumimgs/", img_data['file'])
                if os.path.exists(image_path):
                    with img_cols[idx % num_img_cols]:
                        description = img_data.get('description', '')
                        location = img_data.get('location', '')
                        author = img_data.get('author', '')
                        license = img_data.get('license', '')
                        source_link = img_data.get('source_link', '')
                        attribution = img_data.get('attribution', '') # Fallback
                     
                        # Build caption: combine all available fields
                        caption_parts = []
                        if description:
                            caption_parts.append(description)
                        if location:
                            caption_parts.append(f"Location: {location}")
                        if author:
                            caption_parts.append(f"Author: {author}")
                        if license:
                            caption_parts.append(f"License: {license}")
                        if source_link:
                            caption_parts.append(f"Source: {source_link}")
                        if attribution and not (author or license or source_link): # Use full if fields missing
                            caption_parts.append(attribution)
                        else:
                            caption_parts.append("No additional details available")
                        caption = " | ".join(caption_parts)
                     
                        st.image(image_path, use_container_width=True, caption=caption)
                else:
                    st.warning(f"Image not found: {img_data['file']}")
 
    if st.button("Change Phylum"):
        st.session_state.selected_phylum = None
        st.rerun()
if not selected_phylum:
    st.info("Select a phylum to explore evolutionary patterns.")
else:
    st.info("Now click near a coastline or highlighted hotspot for phylum-specific data.")
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
    m = folium.Map(location=[0, 0], zoom_start=2, tiles=None, attributionControl=False)
    folium.TileLayer("OpenStreetMap", name="Base Map").add_to(m)
    gebco_wms = "https://www.gebco.net/data_and_products/gebco_web_services/web_map_service/mapserv?"
    folium.WmsTileLayer(
        url=gebco_wms,
        layers="GEBCO_LATEST",
        fmt="image/png",
        transparent=True,
        name="Seafloor Bathymetry (GEBCO)",
    ).add_to(m)
    # Add GBIF density layer
    taxon_id = phylum_info[selected_phylum]['gbif_taxon_id']
    density_url = (
        f"https://api.gbif.org/v2/map/occurrence/density/{{z}}/{{x}}/{{y}}@1x.png?"
        f"source=density&taxonKey={taxon_id}&bin=hex&hexPerTile=30&style=classic.poly&srs=EPSG:3857"
    )
    density_layer = folium.TileLayer(
        tiles=density_url,
        attr='Occurrence data from GBIF | © OpenMapTiles',
        name=f"{selected_phylum} Density (GBIF)",
        overlay=True,
        control=True,
        opacity=0.7, # Semi-transparent to see base layers
    ).add_to(m)
    # Add custom attribution control at bottom-left
    AttributionControl().add_to(m)
    # Add scale (measure control) 3D user defined points, click to finish
    # MeasureControl(position='bottomleft', primary_length_unit='kilometers', secondary_length_unit=None, primary_area_unit=None, secondary_area_unit=None).add_to(m)
    
    ScaleControl().add_to(m)
    # m.get_root().header.add_child(folium.Element('<style>.leaflet-bottom .leaflet-control-scale { transform: translateY(-20px); }</style>'))
    m.get_root().header.add_child(folium.Element('<style>.leaflet-control-attribution { background: transparent !important; }</style>'))


    # Hardcoded hotspots (replace with json load if file available)
    hotspots = {
        "Acanthocephala": [
            {"lat": 44.65, "lon": -63.57, "region": "Halifax, Canada (Lobster Habitats)"},
            {"lat": 22.63, "lon": 120.27, "region": "Kaohsiung, Taiwan (Red Snapper Habitats)"}
        ],
    }
    phylum_points = phylum_info[selected_phylum]['hotspots']
    hotspot_layer = folium.FeatureGroup(name=f"{selected_phylum} Hotspots").add_to(m)
    for point in phylum_points:
        folium.Marker(
            [point["lat"], point["lon"]],
            popup=point["region"],
            icon=folium.Icon(color='green', icon='star')
        ).add_to(hotspot_layer)
    # Optional fallback to generic if few hotspots
    if len(phylum_points) < 3:
        generic_points = [
            {"lat": -16.5, "lon": 145.5, "region": "Great Barrier Reef"},
            {"lat": -0.5, "lon": -90.5, "region": "Galápagos Islands"},
            {"lat": 30.0, "lon": -60.0, "region": "Sargasso Sea"},
            {"lat": -55.0, "lon": -60.0, "region": "Antarctic Peninsula"},
            {"lat": 36.8, "lon": -122.0, "region": "Monterey Bay Kelp Forests"}
        ]
        for point in generic_points:
            folium.Marker(
                [point["lat"], point["lon"]],
                popup=f"Generic Marine Hotspot: {point['region']}",
                icon=folium.Icon(color='lightgreen', icon='info-sign')
            ).add_to(hotspot_layer)
        if len(phylum_points) == 0:
            st.info("No specific hotspots for this phylum; showing generic marine hotspots as fallback. Click near them for data.")
    
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
    def fetch_obis_data(geom, size=500, phylum=None):
        occ_list = []
        taxa = []
        fallback_used = False  # New: Flag to track if GBIF fallback was used
        try:
            print(f"Fetching OBIS data for geometry: {geom} with size={size} scientificname={phylum}")
            query = occurrences.search(geometry=geom, size=size, scientificname=phylum)
            occ_data = query.execute()
            if isinstance(occ_data, pd.DataFrame):
                results = occ_data.to_dict('records')
            else:
                results = occ_data.get('results', [])
            print(f'\n\n\nstart test\n\n\n')
            print(f'occ_data\n\n{occ_data}\n\n')
            print(f"Raw OBIS results count: {len(results)}")
            print(f"Total from API: {occ_data.get('total', 0)}") # Check overall matching count
            print(f"First two results: {results[:2]}")
            # Check phyla
            phyla = set(rec.get('phylum') for rec in results if rec.get('phylum'))
            print(f"Unique phyla in results: {phyla}")
            if phyla and all(p.lower() == phylum.lower() for p in phyla):
                print("Success: All results restricted to the phylum!")
            else:
                print("Issue: Outer phyla or mismatches detected.")
           
            # Optional: Print sample records for inspection
            for rec in results[:3]:
                print(f"Sample: scientificName={rec.get('scientificName')}, phylum={rec.get('phylum')}")
           
           
           
            print(f'stop test\n\n\n')
            # Extract all scientific names
            for rec in results:
                if rec: # Skip empty
                    rec_phylum = rec.get('phylum')
                    print(f"Record phylum: {rec_phylum}")
                    sci_name = rec.get('scientificName', 'Unknown')
                    print(f"Adding sci_name: {sci_name}")
                    if sci_name and sci_name != 'Unknown' and isinstance(sci_name, str):
                        taxa.append(sci_name.strip())
             
                    # For occurrences, use lat/lon and the resolved name
                    lat = rec.get('decimalLatitude')
                    lon = rec.get('decimalLongitude')
                    if lat is not None and lon is not None and isinstance(lat, (int, float)) and isinstance(lon, (int, float)):
                        occ_list.append({'lat': lat, 'lon': lon, 'name': sci_name})
            taxa = list(set(taxa)) # Dedupe and clean
            
            # New: GBIF fallback if OBIS taxa < 5
            if len(taxa) < 50:
                fallback_used = True  # Set flag
                print("\n\nFalling back to GBIF due to sparse OBIS data")
                gbif_data = {}
                phylum_key = phylum_info[phylum]['gbif_taxon_id']
                params = {'limit': 500, 'offset': 0}
                if phylum_key:
                    params['phylumKey'] = phylum_key
                else:
                    print("Skipping phylum filter due to key fetch failure")
                
                geom_encoded = urllib.parse.quote(geom)
                query_str = urllib.parse.urlencode(params) + "&geometry=" + geom_encoded
                full_url = base_url + "?" + query_str
                try:
                    print(f'\n\nfull_url:{full_url}\n\n')
                    response = requests.get(full_url)
                    response.raise_for_status()
                    gbif_data = response.json()
                except Exception as e:
                    print(f"GBIF API error: {e}")
                gbif_results = gbif_data.get('results', [])
                # Parse similar to OBIS
                for rec in gbif_results:
                    sci_name = rec.get('scientificName', 'Unknown')
                    if sci_name and sci_name != 'Unknown':
                        taxa.append(sci_name.strip())
                    lat = rec.get('decimalLatitude')
                    lon = rec.get('decimalLongitude')
                    if lat is not None and lon is not None:
                        occ_list.append({'lat': lat, 'lon': lon, 'name': sci_name})
                taxa = list(set(taxa))  # Dedupe after merge
            
                        
            if not taxa:
                print("No taxa found.")
                return {'species': [], 'occurrences': [], 'fallback_used': fallback_used}  # Updated return
            
            print(f"Resolved to {len(taxa)} taxa names: {taxa[:5]}...")
            return {'species': taxa, 'occurrences': occ_list, 'fallback_used': fallback_used}  # Updated return
        except Exception as e:
            print(f"OBIS API error: {e}")
            return {'species': [], 'occurrences': [], 'fallback_used': fallback_used}  # Updated return
      


    def geocode(location):
        url = f"https://nominatim.openstreetmap.org/search?q={location}&format=json"
        headers = {'User-Agent': 'StreamlitOceanExplorer/1.0'}
        response = requests.get(url, headers=headers)
        if response.ok:
            data = response.json()
            if data:
                return float(data[0]['lat']), float(data[0]['lon'])
        return None
    @lru_cache(maxsize=1000)
    @retry(stop=stop_after_attempt(3), wait=wait_fixed(2))
    def fetch_sequence(taxon):
        """Fetch a single COI sequence for a given taxon from NCBI."""
        term = f"{taxon}[Organism] AND COI[Gene Name] AND (\"500\"[SLEN] : \"2000\"[SLEN])"
        print(f"Searching NCBI for: {term}")
        try:
            search_handle = Entrez.esearch(db="nucleotide", term=term, retmax=1, idtype="acc")
            search_results = Entrez.read(search_handle)
            search_handle.close()
            id_list = search_results["IdList"]
            if id_list:
                fetch_handle = Entrez.efetch(db="nucleotide", id=id_list[0], rettype="fasta", retmode="text")
                record = SeqIO.read(fetch_handle, "fasta")
                fetch_handle.close()
                print(f"Fetched sequence for {taxon}: {record.id}")
                return record
            print(f"No sequence found for {taxon}")
            return None
        except Exception as e:
            print(f"Failed to fetch sequence for {taxon}: {e}")
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
    # Updated align_sequences function
    def align_sequences(sequences):
        """Align sequences using MAFFT, with fallback to pairwise alignment or unaligned sequences."""
        try:
            result = subprocess.run(["mafft", "--version"], capture_output=True, text=True, check=False)
            if result.returncode != 0:
                raise FileNotFoundError("MAFFT is not installed or not found in PATH.")
    
            temp_fasta = "temp.fasta"
            aligned_fasta = "aligned.fasta"
    
            # Temporarily shorten IDs to avoid MAFFT truncation issues (>250 chars)
            original_ids = [rec.id for rec in sequences] # Save originals
            for i, rec in enumerate(sequences):
                short_id = f"seq_{i:04d}"
                rec.id = short_id
                rec.name = short_id
                rec.description = short_id
    
            with open(temp_fasta, "w") as f:
                SeqIO.write(sequences, f, "fasta")
    
            subprocess.run(["mafft", "--auto", "--quiet", temp_fasta], stdout=open(aligned_fasta, "w"), check=True, text=True)
    
            if not os.path.exists(aligned_fasta) or os.path.getsize(aligned_fasta) == 0:
                raise RuntimeError("MAFFT produced no output or an empty file.")
    
            aligned = list(SeqIO.parse(aligned_fasta, "fasta"))
    
            # Remap original IDs (order preserved)
            for i, rec in enumerate(aligned):
                rec.id = original_ids[i]
                rec.name = original_ids[i]
                rec.description = original_ids[i]
    
            for file in [temp_fasta, aligned_fasta]:
                if os.path.exists(file):
                    os.remove(file)
    
            if not aligned:
                raise ValueError("No sequences found in MAFFT output.")
    
            print("Alignment successful with MAFFT")
            return aligned
        except FileNotFoundError:
            st.warning("MAFFT is not installed. Falling back to progressive pairwise alignment.")
            print("MAFFT not found. Attempting pairwise alignment.")
            try:
                aligner = PairwiseAligner()
                if len(sequences) < 2:
                    st.error("Insufficient sequences for alignment.")
                    return sequences
        
                # Progressive pairwise for >2 sequences (simple chain)
                aligned_seqs = [sequences[0]]
                for seq in sequences[1:]:
                    alignments = aligner.align(aligned_seqs[-1].seq, seq.seq)
                    # Take first alignment, update last with gaps if needed (basic)
                    aligned_seqs[-1].seq = alignments[0][0] # Update previous
                    seq.seq = alignments[0][1] # Update current
                    aligned_seqs.append(seq)
        
                print("Pairwise alignment successful")
                return aligned_seqs
            except Exception as e:
                st.error(f"Pairwise alignment failed: {e}. Returning unaligned sequences.")
                return sequences
        except Exception as e:
            st.warning(f"MAFFT alignment failed: {e}. Returning unaligned sequences.")
            print(f"MAFFT alignment error: {e}")
            for file in [temp_fasta, aligned_fasta]:
                if os.path.exists(file):
                    os.remove(file)
            return sequences
    def render_tree(newick, title):
        """Render a phylogenetic tree from a Newick string."""
        try:
            tree_io = io.StringIO(newick)
            tree = Phylo.read(tree_io, "newick")
            num_leaves = len(tree.get_terminals())
            fig, ax = plt.subplots(figsize=(8, max(6, num_leaves * 0.5)))
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
    def render_tree_with_images(newick, title, sci_to_label):
        tree_io = io.StringIO(newick)
        tree = Phylo.read(tree_io, "newick")
        num_leaves = len(tree.get_terminals())
        fig = plt.figure(figsize=(12, max(6, num_leaves * 0.5)))
        ax_tree = fig.add_axes([0.05, 0.05, 0.65, 0.9])
        Phylo.draw(tree, axes=ax_tree, do_show=False)
        ax_tree.set_title(title)
        # Extract leaf y positions from text labels
        leaf_y = {}
        for text in ax_tree.texts:
            name = text.get_text()
            leaf_y[name] = text.get_position()[1]
        # Create inverse map for taxon lookup
        label_to_sci = {v: k for k, v in sci_to_label.items()}
        # Add image axes
        ax_images = fig.add_axes([0.72, 0.05, 0.2, 0.9], frameon=False)
        ax_images.set_xlim(-0.1, 1)
        ax_images.set_ylim(ax_tree.get_ylim())
        ax_images.set_xticks([])
        ax_images.set_yticks([])
        # Place images next to leaves
        for label, y in leaf_y.items():
            taxon = label_to_sci.get(label)
            if taxon:
                img_path = get_species_image(taxon)
                if img_path:
                    try:
                        img = Image.open(img_path)
                        # Resize to fixed height for uniform size
                        target_height = 30 # pixels
                        width, height = img.size
                        scale = target_height / height
                        new_size = (int(width * scale), target_height)
                        img = img.resize(new_size, Image.LANCZOS)
                        imagebox = OffsetImage(img, zoom=1)
                        ab = AnnotationBbox(imagebox, (0, y), xycoords='data', boxcoords="data", pad=0, frameon=False, box_alignment=(0, 0.5))
                        ax_images.add_artist(ab)
                    except Exception as e:
                        print(f"Failed to add image for {taxon}: {e}")
        img_buffer = io.BytesIO()
        plt.savefig(img_buffer, format="png", dpi=300)
        img_buffer.seek(0)
        plt.close(fig)
        return Image.open(img_buffer), leaf_y, label_to_sci

    def circle_to_polygon(lon, lat, radius_km=100, num_points=32):
        points = []
        earth_radius = 6371
        for i in range(num_points):
            angle = 2 * math.pi * i / num_points
            dlat = (radius_km / earth_radius) * (180 / math.pi) * math.cos(angle)
            dlon = (radius_km / earth_radius) * (180 / math.pi) / math.cos(lat * math.pi / 180) * math.sin(angle)
            points.append((lon + dlon, lat + dlat))
        points = points[::-1]  # Reverse for counter-clockwise winding
        points.append(points[0])
        wkt = "POLYGON((" + ", ".join(f"{x:.6f} {y:.6f}" for x, y in points) + "))"
        return wkt

    @lru_cache(maxsize=1000)
    def fetch_colloquial_name(taxon):
        """
        Fetch colloquial name with multi-source fallback. Prioritizes English/marine-relevant names.
        Returns (name, source) where name is 'Unknown' if nothing found; italicizes if from a related/fallback source.
        """
        if not taxon or taxon.lower() == 'unknown':
            return 'Unknown', 'N/A'
        if len(taxon.split()) < 2: # Skip non-species, but try phylum-level
            return f"{taxon} (genus or higher)", 'Taxonomy'
        # 1. WoRMS (existing)
        try:
            resp = pyworms.aphiaRecordsByName(taxon, marine_only=False)
            if resp and isinstance(resp, list) and len(resp) > 0:
                vernacular = resp[0].get('vernacularname', '')
                if vernacular:
                    return vernacular.split(',')[0].strip(), 'WoRMS'
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
                            return v['vernacularName'].split(',')[0].strip(), 'GBIF'
                    return results[0].get('vernacularName', 'Unknown'), 'GBIF' # Fallback any
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
                                return v['vernacularName'].split(',')[0].strip() + ' (phylum-level)', 'GBIF'
        except Exception as e:
            print(f"GBIF error for {taxon}: {e}")
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
                                return cn['commonName'].strip(), 'ITIS'
                        return common_names[0].get('commonName', 'Unknown'), 'ITIS' # Fallback
        except Exception as e:
            print(f"ITIS error for {taxon}: {e}")
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
                match = re.search(r'(?:known as|called|common name[:\s]+)(\w+\s+\w+)', abstract, re.IGNORECASE)
                if match:
                    return match.group(1).strip(), 'NCBI Literature'
        except Exception as e:
            print(f"NCBI error for {taxon}: {e}")
        # 5. DuckDuckGo fallback: Instant answer API for quick web search
        try:
            ddg_url = f"https://api.duckduckgo.com/?q=common+name+of+{taxon.replace(' ', '+')}&format=json&pretty=1"
            ddg_resp = requests.get(ddg_url, timeout=5).json()
            abstract = ddg_resp.get('Abstract', '')
            if abstract:
                match = re.search(r'(?:known as|called|common name is|also known as)\s*(?:the\s*)?([\w\s]+?)(?:,|\.|$)', abstract, re.IGNORECASE)
                if match:
                    return match.group(1).strip(), 'DuckDuckGo'
            related = ddg_resp.get('RelatedTopics', [])
            if related:
                for rel in related:
                    text = rel.get('Text', '')
                    match = re.search(r'(?:common name|vernacular)\s*:\s*([\w\s]+)', text, re.IGNORECASE)
                    if match:
                        return match.group(1).strip(), 'DuckDuckGo'
        except Exception as e:
            print(f"DuckDuckGo error for {taxon}: {e}")
        return 'Unknown', 'N/A'
  
    @lru_cache(maxsize=100) # Cache to avoid redundant OBIS calls
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
        for taxon in set(taxon_list): # Dedupe
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
                if not results or not isinstance(results, list):
                    print(f"No valid results for {taxon} in checklist")
                    continue
                if not results[0]:
                    print(f"results[0] is None or falsy for {taxon}")
                    continue
                aphiaid = results[0].get('aphiaID')
                if not aphiaid:
                    print(f"No AphiaID for {taxon}")
                    continue
      
                # Get child species via OBIS API
                child_url = f"https://api.obis.org/v3/taxon/{aphiaid}/children?rank=Species"
                response = requests.get(child_url)
                response.raise_for_status()
                child_data = response.json()
                child_results = child_data.get('results', [])
                if not child_results or not isinstance(child_results, list):
                    print(f"No valid child results for {taxon}")
                    continue
                child_species = [rec['scientificName'] for rec in child_results if rec.get('taxonRank') == 'Species']
      
                if not child_species:
                    print(f"No child species for {taxon}")
                    continue
      
                # Filter by proximity (bounding box: ±5° around center)
                bbox = f"POLYGON(({center_lon-5} {center_lat-5},{center_lon-5} {center_lat+5},{center_lon+5} {center_lat+5},{center_lon+5} {center_lat-5},{center_lon-5} {center_lat-5}))"
                geo_filtered = []
                for sp in child_species[:10]: # Limit to avoid API overload
                    sp_occ = occurrences.search(scientificname=sp, geometry=bbox, size=1).execute()
                    if sp_occ.get('results') or (isinstance(sp_occ, pd.DataFrame) and not sp_occ.empty):
                        geo_filtered.append(sp)
                        if len(geo_filtered) >= max_per_taxon:
                            break
      
                # If no geo-match, pick first species as fallback
                to_add = geo_filtered if geo_filtered else child_species[:max_per_taxon]
                resolved.extend(to_add)
                print(f"Resolved {taxon} to: {to_add}")
            except Exception as e:
                print(f"Resolution failed for {taxon}: {e}")
        resolved = list(set(resolved)) # Dedupe final list
        if not resolved:
            print("No species resolved; returning original list")
            return [t for t in taxon_list if isinstance(t, str) and len(t.split()) >= 2] # Keep only species-like
        return resolved
    def get_species_image(taxon):
        """Fetch a thumbnail image URL for the species from WoRMS or Wikimedia Commons, save locally."""
        colloquial, _ = fetch_colloquial_name(taxon)
        search_term = colloquial if colloquial != 'Unknown' else taxon
        img_url = None
        # Prioritize WoRMS
        try:
            resp = pyworms.aphiaRecordsByName(taxon, marine_only=False)
            if resp and isinstance(resp, list) and len(resp) > 0:
                aphia_id = resp[0].get('AphiaID')
                if aphia_id:
                    worms_url = f"https://www.marinespecies.org/aphia.php?p=taxdetails&id={aphia_id}"
                    response = requests.get(worms_url)
                    if response.ok:
                        html = response.text
                        # Adjusted regex to match the gallery class
                        match = re.search(r'<div[^>]*class="gallery"[^>]*>.*?<img\s+[^>]*src="([^"]+)"', html, re.S | re.I)
                        if match:
                            img_src = match.group(1)
                            if not img_src.startswith('http'):
                                img_src = 'https://www.marinespecies.org' + img_src
                            img_url = img_src
        except Exception as e:
            print(f"Failed to fetch WoRMS image for {taxon}: {e}")
        # Fallback to Wikimedia Commons search
        if not img_url:
            try:
                search_url = f"https://commons.wikimedia.org/w/api.php?action=query&list=search&srsearch={requests.utils.quote(search_term)}&srnamespace=6&format=json&srlimit=1"
                resp = requests.get(search_url).json()
                search_results = resp['query']['search']
                if search_results:
                    title = requests.utils.quote(search_results[0]['title'])
                    image_url = f"https://commons.wikimedia.org/w/api.php?action=query&titles={title}&prop=imageinfo&iiprop=url&iiurlwidth=200&format=json"
                    img_resp = requests.get(image_url).json()
                    pages = img_resp['query']['pages']
                    for page in pages.values():
                        if 'imageinfo' in page:
                            img_url = page['imageinfo'][0]['thumburl']
            except Exception as e:
                print(f"Failed to fetch Wikimedia Commons image for {search_term}: {e}")
        if img_url:
            # Determine extension
            ext = img_url.split('.')[-1].split('?')[0]
            if len(ext) > 4: # Invalid ext
                ext = 'jpg'
            filename = taxon.replace(' ', '_') + '.' + ext
            img_dir = './data/imgdownloads'
            os.makedirs(img_dir, exist_ok=True)
            filepath = os.path.join(img_dir, filename)
            if not os.path.exists(filepath):
                try:
                    response = requests.get(img_url, timeout=5)
                    if response.ok:
                        with open(filepath, 'wb') as f:
                            f.write(response.content)
                except Exception as e:
                    print(f"Failed to download image for {taxon}: {e}")
                    return None
            return filepath
        return None
    def build_phylogenetic_tree(species_list, num_sequences, region, center_lat=None, center_lon=None):
        """
        Build a phylogenetic tree from COI sequences for given species. Always uses scientific names for labels during construction.
        Args:
            species_list (list): List of taxa from OBIS.
            num_sequences (int): Number of sequences to fetch.
            region (str): Region name for display.
            center_lat (float): Latitude of clicked point (optional).
            center_lon (float): Longitude of clicked point (optional).
        Returns:
            tuple: (newick string with scientific labels, phylogenetic diversity score, insight message with scientific names, used_taxa)
        """
        if not species_list or all(s == 'Unknown' for s in species_list):
            st.warning(f"No valid species found for {region}. Cannot build phylogenetic tree.")
            return None, 0.0, "No valid species data available.", []
        # Resolve higher ranks to species, using geographic context if provided
        if center_lat is not None and center_lon is not None:
            with st.spinner("Resolving higher-rank taxa to species..."):
                species_list = resolve_to_species(tuple(species_list), center_lat, center_lon, max_per_taxon=1)
                if not species_list:
                    st.warning(f"No species-level taxa resolved for {region}. Using original list.")
                    species_list = [s for s in species_list if s != 'Unknown']
                else:
                    st.info(f"Resolved to {len(species_list)} species-level taxa for {region}.")
                    print(f"Resolved species: {species_list[:5]}...")
        valid_species = [s for s in species_list if s != 'Unknown' and is_species_level(s)]
        if not valid_species:
            valid_species = [s for s in species_list if s != 'Unknown']
        if not valid_species:
            st.warning(f"No valid taxa found for {region}. Cannot build phylogenetic tree.")
            return None, 0.0, "No valid taxa data available.", []
        sequences = []
        used_taxa = []
        failed_taxa = []
        progress_bar = st.progress(0)
        target_sequences = min(num_sequences, len(valid_species))
        for i, taxon in enumerate(valid_species[:target_sequences * 10]):
            try:
                record = fetch_sequence(taxon)
                if record:
                    colloquial, source = fetch_colloquial_name(taxon)
                    ncbi_id = record.id
                    label = taxon # Always use scientific name for building
                    record.id = label
                    record.name = label
                    sequences.append(record)
                    used_taxa.append((taxon, colloquial, ncbi_id))
                else:
                    failed_taxa.append(taxon)
                if len(sequences) >= target_sequences:
                    break
                progress_bar.progress(min((i + 1) / (target_sequences * 10), 1.0))
            except Exception as seq_e:
                print(f"Failed to fetch sequence for {taxon}: {seq_e}")
                failed_taxa.append(taxon)
        progress_bar.empty()
        if failed_taxa and len(sequences) < target_sequences:
            st.warning(
                f"Could not fetch sequences for {len(failed_taxa)} taxa (e.g., {', '.join(failed_taxa[:3])}). "
                f"Only {len(sequences)} sequences retrieved."
            )
        if len(sequences) < 2:
            st.warning(f"Insufficient sequences ({len(sequences)}) to build a tree for {region}.")
            return None, 0.0, f"Only {len(sequences)} sequences retrieved; minimum 2 required.", []
        try:
            with st.spinner("Aligning sequences and building tree..."):
                aligned_sequences = align_sequences(sequences)
                aln = MultipleSeqAlignment(aligned_sequences)
        
                # Debug: Check for duplicates before distance calc
                names = [s.id for s in aln]
                unique_names = set(names)
                if len(names) != len(unique_names):
                    st.error(f"Duplicates detected after alignment for {region}: {[n for n in unique_names if names.count(n) > 1]}")
                    raise ValueError("Duplicate names found after alignment")
        
                calculator = DistanceCalculator('identity')
                dm = calculator.get_distance(aln)
                constructor = DistanceTreeConstructor()
                tree = constructor.nj(dm)
                output = io.StringIO()
                Phylo.write(tree, output, 'newick')
                newick = output.getvalue().strip()
                dtree = dendropy.Tree.get(data=newick, schema='newick')
                pd_score = sum(e.length for e in dtree.edges() if e.length is not None)
                pd_score = 0.0 if math.isnan(pd_score) or pd_score is None else pd_score # Handle NaN
                divergence_insight = f"Tree built from COI sequences of {', '.join([s.name for s in sequences])}"
                print(f"Built tree with PD score: {pd_score}")
                return newick, pd_score, divergence_insight, used_taxa
        except Exception as e:
            st.warning(f"Tree construction failed for {region}: {e}.")
            return None, 0.0, f"Tree construction failed: {str(e)}", []
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
    rarity = phylum_info[selected_phylum]['rarity']
    if rarity == 'Common':
        radius_km = 100
        expand_steps = [200, 500] # Smaller expansions
    elif rarity == 'Medium':
        radius_km = 500
        expand_steps = [800, 1200]
    else: # Rare
        radius_km = 1000
        expand_steps = [1300, 2000]
    st.info(f"Phylum: {selected_phylum} ({rarity}). Search radius: {radius_km}km")
    map_output = st_folium(m, width=700, height=500, returned_objects=["last_clicked"], key=f"main_map_{st.session_state.click_counter}")
    # Handle map clicks
    if map_output and map_output.get("last_clicked"):
        st.session_state.click_counter += 1
        clicked_lat = map_output["last_clicked"]["lat"]
        clicked_lon = map_output["last_clicked"]["lng"]
        st.write(f"Clicked location: Lat {clicked_lat:.2f}, Lon {clicked_lon:.2f}")
        print(f"Map click detected: Lat {clicked_lat}, Lon {clicked_lon}")
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
        # Zoom to the clicked area
        bounds = poly_gdf.total_bounds
        m.fit_bounds([[bounds[1], bounds[0]], [bounds[3], bounds[2]]])
        with st.spinner("Fetching OBIS data..."):
            obis_data = fetch_obis_data(geom, size=500, phylum=selected_phylum)
            if obis_data.get('fallback_used'):
                st.info("Sparse OBIS data; including GBIF results for better coverage.")
            species_list = obis_data['species']
            occ_list = obis_data['occurrences']
            # Dynamic radius expansion if sparse
            expanded_radii = expand_steps
            for exp_radius in expanded_radii:
                if len(species_list) >= 5:
                    break
                new_radius = exp_radius  # Set to absolute value from steps
                st.info(f"Sparse data at {radius_km}km; auto-expanding to {new_radius}km.")
                geom = circle_to_polygon(clicked_lon, clicked_lat, radius_km=new_radius)
                obis_data = fetch_obis_data(geom, size=500, phylum=selected_phylum)
                species_list = obis_data['species']
                occ_list = obis_data['occurrences']
                poly_gdf = gpd.GeoDataFrame(
                    {"name": ["Expanded Search Area"]},
                    geometry=[loads(geom)],
                    crs="EPSG:4326"
                    )
                folium.GeoJson(
                    poly_gdf,
                    name="Expanded Search Polygon",
                    style_function=lambda x: {"color": "purple", "weight": 2, "fillOpacity": 0.1},
                    tooltip=f"Expanded Search Area (~{new_radius}km radius)"
                    ).add_to(click_layer)
                radius_km = new_radius # Update for display
            pd_score = 0.0
            if species_list:
                with st.expander("Species at Clicked Location"):
                    species_data = []
                    for s in species_list:
                        colloquial, source = fetch_colloquial_name(s) if s != 'Unknown' else ('Unknown', 'N/A')
                        status = "Direct from OBIS" if len(s.split()) >= 2 else "Resolved via OBIS API"
                        species_data.append({"Scientific Name": s, "Common Name": colloquial, "Common Name Source": source, "Resolution Status": status})
                    st.write(f"Found {len(species_list)} species")
                    st.dataframe(pd.DataFrame(species_data))
    
                if len(species_list) <= 1:
                    num_sequences = 1 if len(species_list) == 1 else 0
                    st.info(f"Only {len(species_list)} species available. Setting number of sequences to {num_sequences}.")
                else:
                    num_sequences = st.slider(
                        "Number of sequences to fetch for tree (COI)",
                        min_value=1,
                        max_value=min(100, len(species_list)),
                        value=min(10, len(species_list)),
                        key=f"num_sequences_{st.session_state.click_counter}"
                    )
        

                label_type = st.selectbox(
                    "Tree Node Labels",
                    ["Scientific Name", "Common Name", "NCBI Accession"],
                    key=f"label_type_{st.session_state.click_counter}"
                )
                if num_sequences > 20:
                    st.warning(f"Fetching {num_sequences} sequences may take several minutes. Please be patient.")
                tree_key = f"tree_base_{st.session_state.click_counter}"
                if tree_key not in st.session_state:
                    base_newick, pd_score, base_insight, used_taxa = build_phylogenetic_tree(
                        species_list, num_sequences, "Clicked Location", clicked_lat, clicked_lon
                    )
                    st.session_state[tree_key] = {
                        "base_newick": base_newick,
                        "pd_score": pd_score,
                        "used_taxa": used_taxa
                    }
                else:
                    cached = st.session_state[tree_key]
                    base_newick = cached["base_newick"]
                    pd_score = cached["pd_score"]
                    used_taxa = cached["used_taxa"]
                if base_newick:
                    # Load base tree (with scientific names)
                    tree_io = io.StringIO(base_newick)
                    tree = Phylo.read(tree_io, "newick")
                    # Get leaf order based on scientific names
                    leaf_order = [term.name for term in tree.get_terminals()]
                    # Create mapping from scientific to selected label
                    sci_to_label = {}
                    for taxon, colloquial, ncbi_id in used_taxa:
                        if label_type == "Common Name":
                            label = f"{colloquial} ({taxon})" if colloquial != 'Unknown' else taxon
                        elif label_type == "NCBI Accession":
                            label = ncbi_id
                        else:
                            label = taxon
                        sci_to_label[taxon] = label
                    # Relabel terminals
                    for term in tree.get_terminals():
                        sci_name = term.name
                        if sci_name in sci_to_label:
                            term.name = sci_to_label[sci_name]
                    # Write new newick
                    output = io.StringIO()
                    Phylo.write(tree, output, 'newick')
                    newick = output.getvalue().strip()
                    # Update insight with new labels
                    divergence_insight = f"Tree built from COI sequences of {', '.join([term.name for term in tree.get_terminals()])}"
                    label_to_sci = {v: k for k, v in sci_to_label.items()}
                    tree_img, leaf_y, label_to_sci = render_tree_with_images(newick, "Phylogenetic Tree at Clicked Location", sci_to_label)
                    st.write(f"Phylogenetic Diversity: {pd_score:.1f}")
                    st.write(divergence_insight)
                    if tree_img:
                        st.image(tree_img)
                        # Display larger images below
                        st.subheader("Larger Species Images (Click browser zoom or right-click to enlarge)")
                        num_img_cols = 3
                        img_cols = st.columns(num_img_cols)
                        for i, (label, y) in enumerate(leaf_y.items()):
                            taxon = label_to_sci.get(label)
                            if taxon:
                                img_path = get_species_image(taxon)
                                if img_path:
                                    with img_cols[i % num_img_cols]:
                                        st.image(img_path, caption=label, width=200)
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
                    obis_data = fetch_obis_data(geom, size=100, phylum=selected_phylum)
                    if obis_data.get('fallback_used'):
                        st.info("Sparse OBIS data; including GBIF results for better coverage.")
                    species_list = obis_data['species']
                else:
                    st.error("Could not geocode location or find species.")
            if species_list:
                with st.expander("Species Found"):
                    species_data = []
                    for s in species_list:
                        colloquial, source = fetch_colloquial_name(s) if s != 'Unknown' else ('Unknown', 'N/A')
                        status = "Direct from OBIS" if len(s.split()) >= 2 else "Resolved via OBIS API"
                        species_data.append({"Scientific Name": s, "Common Name": colloquial, "Common Name Source": source, "Resolution Status": status})
                    st.write(f"Found {len(species_list)} species")
                    st.dataframe(pd.DataFrame(species_data))

                # if we let user select
                # num_sequences = st.slider(
                    # "Number of sequences to fetch (COI)",
                    # min_value=1,
                    # max_value=min(100, len(species_list)),
                    # value=min(10, len(species_list)),
                    # key="search_num_sequences"
                # )

                if len(species_list) <= 1:
                    num_sequences = 1 if len(species_list) == 1 else 0
                    st.info(f"Only {len(species_list)} species available. Setting number of sequences to {num_sequences}.")
                else:
                    num_sequences = st.slider(
                        "Number of sequences to fetch (COI)",
                        min_value=1,
                        max_value=min(100, len(species_list)),
                        value=min(10, len(species_list)),
                        key="search_num_sequences"
                    )

                label_type = st.selectbox(
                    "Tree Node Labels",
                    ["Scientific Name", "Common Name", "NCBI Accession"],
                    key="search_label_type"
                )
                if num_sequences > 20:
                    st.warning(f"Fetching {num_sequences} sequences may take several minutes. Please be patient.")
                # For search, use a simple cache based on search term
                if 'last_search' not in st.session_state or st.session_state.last_search != search:
                    st.session_state.last_search = search
                    st.session_state.search_tree_base = None
                tree_key = "search_tree_base"
                if st.session_state.get(tree_key) is None:
                    lat = coord[0] if 'coord' in locals() and coord else None
                    lon = coord[1] if 'coord' in locals() and coord else None
                    base_newick, pd_score, base_insight, used_taxa = build_phylogenetic_tree(
                        species_list, num_sequences, search, lat, lon
                    )
                    st.session_state[tree_key] = {
                        "base_newick": base_newick,
                        "pd_score": pd_score,
                        "used_taxa": used_taxa
                    }
                else:
                    cached = st.session_state[tree_key]
                    base_newick = cached["base_newick"]
                    pd_score = cached["pd_score"]
                    used_taxa = cached["used_taxa"]
                if base_newick:
                    # Load base tree (with scientific names)
                    tree_io = io.StringIO(base_newick)
                    tree = Phylo.read(tree_io, "newick")
                    # Get leaf order based on scientific names
                    leaf_order = [term.name for term in tree.get_terminals()]
                    # Create mapping from scientific to selected label
                    sci_to_label = {}
                    for taxon, colloquial, ncbi_id in used_taxa:
                        if label_type == "Common Name":
                            label = f"{colloquial} ({taxon})" if colloquial != 'Unknown' else taxon
                        elif label_type == "NCBI Accession":
                            label = ncbi_id
                        else:
                            label = taxon
                        sci_to_label[taxon] = label
                    # Relabel terminals
                    for term in tree.get_terminals():
                        sci_name = term.name
                        if sci_name in sci_to_label:
                            term.name = sci_to_label[sci_name]
                    # Write new newick
                    output = io.StringIO()
                    Phylo.write(tree, output, 'newick')
                    newick = output.getvalue().strip()
                    # Update insight with new labels
                    divergence_insight = f"Tree built from COI sequences of {', '.join([term.name for term in tree.get_terminals()])}"
                    label_to_sci = {v: k for k, v in sci_to_label.items()}
                    tree_img, leaf_y, label_to_sci = render_tree_with_images(newick, f"Evolution for {search}", sci_to_label)
                    st.write(f"Species Count: {len(species_list)}")
                    st.write(f"Phylogenetic Diversity (PD): {pd_score:.1f}")
                    st.markdown(f"**Evolutionary Insight**: {divergence_insight}")
                    if tree_img:
                        st.image(tree_img, caption=f"Phylogenetic tree showing evolutionary flow for {search}")
                        # Display larger images below
                        st.subheader("Larger Species Images (Click browser zoom or right-click to enlarge)")
                        num_img_cols = 3
                        img_cols = st.columns(num_img_cols)
                        for i, (label, y) in enumerate(leaf_y.items()):
                            taxon = label_to_sci.get(label)
                            if taxon:
                                img_path = get_species_image(taxon)
                                if img_path:
                                    with img_cols[i % num_img_cols]:
                                        st.image(img_path, caption=label, width=200)
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
                                colloquial, source = fetch_colloquial_name(taxon)
                                record.name = f"{taxon} ({colloquial})"
                                sequences.append(record)
                            else:
                                failed_taxa.append(taxon)
                            if len(sequences) >= num_sequences:
                                break
                            progress_bar.progress(min((i + 1) / (num_sequences * 10), 1.0))
                        except Exception as seq_e:
                            print(f"Failed to fetch sequence for {taxon}: {seq_e}")
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
This app emphasizes phylogenetic diversity (PD) for marine conservation, aligning with a 2025 Nature study on preserving evolutionary potential in ocean ecosystems. Incorporating eDNA methods, as seen in 2025 research on fish communities, as seen in recent fungal diversity studies. TimeTree integration adds temporal depth, reflecting research on marine PD changes. This app (API loads in ~5-10s) balances interactivity and data-driven bioinformatics.
""")
