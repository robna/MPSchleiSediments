# gepardBlindCorr

This repository contains Python scripts and Jupyter notebooks for microplastic data processing with GEPARD blind correction procedures. It is a focused fork of the MPSchleiSediments repository, containing only the essential MPDB (Microplastic Database) processing scripts.

## What is this project about?

This project provides tools for processing microplastic particle data with quality control procedures including:
- Blank correction for laboratory contamination removal  
- Blind sample processing for method validation
- Particle size filtering and standardization
- Data export for further analysis

## Repository Contents

### Core Python Modules

**`MPDB_utils.py`**
- Utility functions for data preprocessing
- Particle amplification based on analyzed fractions
- Size filtering and geometric mean calculations
- Shape and color standardization
- Data separation into environmental, blind, and blank samples

**`MPDB_procedures.py`**  
- Blank procedure for contamination removal
- Blind procedure for method validation
- Synthetic blind sample generation
- Particle elimination tracking and logging

**`MPDB_settings.py`**
- Database connection configuration
- SQL query definitions
- Polymer exclusion lists
- Blind sample configurations
- Size filtering parameters

### Jupyter Notebook

**`MPDB_notebook.ipynb`**
- Complete data processing pipeline
- Database connection and data extraction
- Step-by-step application of all processing procedures
- Data export functionality

## Key Features

**Quality Control Pipeline:**
- Laboratory blank correction removes contamination artifacts
- Procedural blind validation ensures method reliability  
- Configurable polymer exclusion lists
- Size-based particle filtering

**Data Processing:**
- Particle amplification for sub-sample extrapolation
- Geometric mean size calculations
- Shape and color standardization
- Automated particle ID management

**Export Capabilities:**
- Clean, processed microplastic datasets
- Elimination tracking for transparency
- CSV export for downstream analysis

## Getting Started

### Prerequisites
- Python 3.8+
- pandas, numpy
- mysql-connector-python (for database access)
- jupyter (for notebook execution)

### Installation
```bash
git clone [repository-url]
cd gepardBlindCorr
pip install pandas numpy mysql-connector-python jupyter
```

### Basic Usage

1. **Configure Database Connection:**
   - Update `MPDB_server` in `MPDB_settings.py` if needed
   - Ensure you have MPDB database access credentials

2. **Run Processing Pipeline:**
   ```python
   # Open and run MPDB_notebook.ipynb
   # Follow the step-by-step processing workflow
   # Provides interactive database connection prompts
   ```

3. **Programmatic Usage:**
   ```python
   import pandas as pd
   from MPDB_utils import *
   from MPDB_procedures import *
   
   # Load and process your data through the pipeline
   # See notebook for complete example
   ```

## Methods

**Blank Correction:**
- Matches particles in samples with laboratory blank particles
- Removes closest size matches by phenotype (polymer, color, shape)
- Tracks all eliminations for transparency

**Blind Validation:**
- Processes synthetic blind samples for method validation
- Creates representative blind particle distributions
- Eliminates corresponding environmental particles

**Quality Filters:**
- Excludes problematic polymers (PTFE, Parafilm, etc.)
- Applies configurable size filters
- Standardizes particle classifications

## Configuration

Key settings in `MPDB_settings.py`:
- `size_filter_highpass`: Minimum particle size (default: 50 μm)
- `size_filter_lowpass`: Maximum particle size (default: 5000 μm)  
- `polyDropList`: Polymers to exclude from analysis
- `blindList`: Blind samples to include in processing

## License

This project is licensed under the GPL-3.0 License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This is a focused fork of the MPSchleiSediments repository, containing the essential microplastic database processing components developed for the BONUS micropollutants project.