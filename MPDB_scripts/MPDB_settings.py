MPDB_server: str = '192.124.245.26'

particleQuery: str = '''-- TODO: simplify column names, e.g. avoid non-ascii characters
                SELECT distinct
                    `p`.`Sample` AS `Sample`,
                    `p`.`IDParticles` AS `IDParticles`,
                    `s`.`Site_name` AS `Site_name`,
                    `s`.`GPS_LON` AS `GPS_LON`,
                    `s`.`GPS_LAT` AS `GPS_LAT`,
                    `s`.`Compartment` AS `Compartment`,
                    `s`.`Contributor` AS `Contributor`,
                    `s`.`Project` AS `Project`,
                    `p`.`Particle_name` AS `Particle_name`,
                    `p`.`Size_1_[µm]` AS `Size_1_[µm]`,
                    `p`.`Size_2_[µm]` AS `Size_2_[µm]`,
                    `p`.`Size_3_[µm]` AS `Size_3_[µm]`,
                    `p`.`Shape` AS `Shape`,
                    `p`.`Colour` AS `Colour`,
                    `p`.`Preferred_method` AS `Preferred_method`,
                    `pt`.`polymer_type` AS `polymer_type`,
                    `a`.`Library_entry` AS `library_entry`,
                    `a`.`Comment` AS `Comment`,
                    `s`.`Lab_blank` AS `lab_blank_ID`,
                    `s`.`IDSample` AS `sample_ID`,
                    `s`.`Sampling_weight_[kg]` AS `Sampling_weight_[kg]`,
                    `s`.`Fraction_analysed` AS `Fraction_analysed`
                FROM ((((`particles` `p`
                    JOIN `samples` `s` ON ((`p`.`Sample` = `s`.`Sample_name`)))
                    JOIN `particles2analysis` `pa` ON ((`p`.`IDParticles` = `pa`.`IDParticles`)))
                    JOIN `analysis` `a` ON ((`pa`.`IDAnalysis` = `a`.`IDAnalysis`)))
                    JOIN `polymer_type` `pt` ON ((`a`.`Result` = `pt`.`IDPolymer_type`)))'''

# List for dropping system caused contamination (PTFE, PV23, Parafilm)
# and certain dyes if they are no distinct indicators for synthetic polymers
polyDropList: list = ['Poly (tetrafluoro ethylene)',
                      'PV23',
                      'Parafilm',
                      'PR101',
                      'PB15',
                      'PW6',
                      'PBr29',
                      'PY17based',
                      'PY74',
                      'PB15 + PV23',
                      'PV23 + PB15',
                      'PB15 + TiO2',
                      'PB23 + PY17based',
                      'Parafilm/PE',
                      'PB15+PY17',
                      'PY17+PB15',
                      'PV23+PB15+TiO2',
                      'PB15+TiO2',
                      'TiO2+PB15',
                      'PB15+PV23']


# TODO: For now this dict is only to have shorter names, i.e. when using blind names in file names of plots. It's not used by the notebook.
blindNames = {'A': 'Blank_11.12.2018_w_IS',
              'B': 'Blank_20.12.2018_w_IS',
              'C': 'Blank_11.02.19',
              'D': 'Blank_5.11.19_IS_1',
              'E': 'Blank_5.11.19_IS_2',
              'F': 'Blank_6.11.19_1',
              'G': 'Blank_6.11.19_2',
              'H': 'Blank_20.11.19',
              'I': 'Blank_20.11.19_IS',
              'J': 'Blank_5.5.21'
}


# blind samples to be used for analysis
blindList = ['Blank_11.12.2018_w_IS',
             'Blank_20.12.2018_w_IS',
             'Blank_11.02.19',
             'Blank_5.11.19_IS_1',
             'Blank_5.11.19_IS_2',
             'Blank_6.11.19_1',
             'Blank_6.11.19_2',
             # 'Blank_20.11.19',  # Julians blanks excluded as they are higher than normal contaminated
             # 'Blank_20.11.19_IS',  # Julians blanks excluded as they are higher than normal contaminated
             'Blank_5.5.21'
             ]


class Config:  # TODO: NEEDS CHECKING: how does this work together with the other size limits set in the analysis/settings.py??
    size_filter_dimension: str = 'Size_1_[µm]'  # Sets the dimension on which to apply the > x µm filter.
    # Can be either 'Size_1_[µm]', 'Size_2_[µm]', or 'size_geom_mean'.
    size_filter_highpass: int = 50  # Value in µm applied to size_filter_dimension to only keep particle >= in size.
    size_filter_lowpass: int = 5000  # Value in µm applied to size_filter_dimension to only keep particle < in size.
