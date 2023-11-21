# MPSchleiSediments ![micropoll_logo](https://www.io-warnemuende.de/files/project/micropoll/images/MICROPOLL_Logo.jpg)

This repository contains code for analyzing microplastics data from Schlei fjord sediments as part of the [BONUS MicroPoll](https://www.io-warnemuende.de/micropoll-home.html) project.

## Project Overview

The [BONUS MicroPoll](https://www.io-warnemuende.de/micropoll-home.html) project aims to assess the impact of microplastics on the Baltic Sea ecosystem. As part of this project, we are analyzing microplastics data from Schlei fjord sediments. The data was collected by the [Leibniz Institute for Baltic Sea Research Warnemünde](https://www.io-warnemuende.de/home.html) (IOW) and is available from the [MPDB database](https://www.io-warnemuende.de/tl_files/project/micropoll/Deliverables/call2015-122_D1.2_0.2.pdf).

The data analysis in this repository is divided into two parts:

1. **Data preparation and cleaning**. The data is loaded from the MPDB database and cleaned. This includes the removal of blanks and blinds, as well as the removal of outliers. The resulting data is saved in the [data folder](data) and used for the analysis in the second part.
2. **Data analysis**. The cleaned data is analyzed and visualized. This includes exploratory data analysis, geospatial analysis, and modelling.

## Project Directory Structure

The project is structured as follows:

```bash
.
│
├── MPDB_scripts
│   ├── MPDB_notebook.ipynb                      # Notebook for querying the MPDB database and performing blank/blind-removal
│   ├── MPDB_procedures.py
│   ├── MPDB_settings.py                         # Central place to manage settings for the MPDB scripts
│   └── MPDB_utils.py
│
├── analysis
│   ├── analysis_geospatial.ipynb                # Notebook for geospatial analysis and interpolation
│   ├── analysis.ipynb                           # Notebook for preliminaty data analysis and specific visualization
│   ├── app_helpers.py
│   ├── app_loaders.py
│   ├── app.py                                  # Streamlit app for interactive data exploration
│   ├── components.py
│   ├── correlations.py
│   ├── cv_helpers.py                           # Functions related to cross-validation
│   ├── cv.ipynb                                # Notebook for modelling and cross-validation
│   ├── cv.py                                   # Functions related to modelling and cross-validation
│   ├── dists.py
│   ├── geo_io.py                               # Functions for input/output of geo data
│   ├── geo.py                                  # Functions for geospatial analysis
│   ├── glm.py
│   ├── helpers.py
│   ├── interpol.py                             # Functions for geospacial interpolation
│   ├── KDE_utils.py
│   ├── outliers.py
│   ├── plots.py                                # Central place for plotting functions
│   ├── prepare_data.py                         # Functions to load, combine and prepare data for analysis
│   └── settings.py                             # Central place to manage settings for the analysis scripts
│   
├── data
│   ├── BAW_tracer_simulations.zip               # Contains tracer simulations for the Schlei fjord
│   ├── GRADISTAT_CAU_vol_log-cau_closed.csv
│   ├── GRADISTAT_IOW_count_log-cau_not-closed.csv
│   ├── GRADISTAT_IOW_vol_log-cau_not-closed.csv
│   ├── ManualHeights_Schlei_S8_v2.csv
│   ├── Metadata_CAU_sampling_log.csv
│   ├── Metadata_IOW_sampling_log.csv
│   ├── model_data.csv
│   ├── mp_pdd.csv                              # Contains cleaned microplastics particle data (output of MPDB_notebook.ipynb)
│   ├── pred_data.csv
│   ├── SchleiCoastline_from_OSM.geojson
│   ├── Schlei_OM.csv
│   ├── sdd_CAU.csv
│   ├── sdd_IOW.csv
│   ├── sediment_grainsize_CAU_vol_log-cau_closed.csv
│   ├── sediment_grainsize_IOW_count_log-cau_not-closed.csv
│   ├── sediment_grainsize_IOW_vol_linear_not-closed.csv
│   ├── sediment_grainsize_IOW_vol_log-cau_not-closed.csv
│   ├── WWTP_influence_CAU.csv
│   ├── WWTP_influence_IOW.csv
│   └── exports
│       ├── exports_README.md
│       ├── cache
│       │   └── cache_README.md
│       ├── geo
│       │   └── geo_README.md
│       ├── models
│       │   ├── models_README.md
│       │   ├── logs
│       │   │   └── logs_README.md
│       │   ├── model_NCV_result.csv
│       │   ├── predictions
│       │   │   └── predictions_README.md
│       │   └── serialised
│       │       └── serialised_README.md
│       └── plots
│           └── plots_README.md
│
│
├── .gitignore
├── Pipfile
├── Pipfile.lock
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Getting Started

To use, reproduce or build upon the data analysis in this repository, follow these steps:

1. Clone the repository to your local machine and set up a working environment with the required packages (see [Requirements](#requirements)).

2. Run the scripts from the `MPDB_scripts` folder
   - Use the `MPDB_notebook.ipynb` notebook to run them and see their output. These scripts load the available data from the [MPDB database](https://www.io-warnemuende.de/tl_files/project/micropoll/Deliverables/call2015-122_D1.2_0.2.pdf) and perform the blank/blind-removal procedure. The resulting (shortened) data is saved in the [data folder](data). From there it will be imported into the analysis scripts.

3. Start the analysis.
   - The `cv.ipynb` notebook in the `analysis` folder contains code for modelling and cross-validation. It uses the data from the [data folder](data) and saves the results in the [exports folder](data/exports).

4. Run the geospatial analysis.
   - The `analysis_geospatial.ipynb` notebook in the `analysis` folder contains code for geospatial analysis and interpolation. It uses the data from the [data folder](data) and saves the results in the [exports folder](data/exports).

5. Run the Streamlit app.
   - The `app.py` script in the `analysis` folder contains code for an interactive data exploration app. It provides a convenient way to explore the data and visualize the results of the analysis. To run the app, execute `streamlit run app.py` from within the `analysis` subdirectory. This will open the app in your browser.

## Requirements

The code in this repository requires Python $\geq$ 3.8 and the packages listed in the requirements file. For convenience, we provide reuqirements files for setting up a working environment using [pipenv](https://pipenv.pypa.io/en/latest/) (use `Pipfile`) or [pdm](https://pdm.fming.dev/) (use `pyproject.toml`). Alternatively, you can install the required packages manually. Or you can set up an environment with a different package manager of your choice (use `requirements.txt` as a reference).


## Contributing

If you would like to contribute to this project, please fork the repository and submit a pull request. We welcome contributions of all kinds, including bug fixes, feature requests, and documentation improvements.

## License

This project is licensed under the <...> License - see the [<...>](LICENSE) file for details.

## Acknowledgments

This project was made possible by the [BONUS MicroPoll](https://www.io-warnemuende.de/micropoll-home.html) project. We would like to thank the project team for their support and contributions.