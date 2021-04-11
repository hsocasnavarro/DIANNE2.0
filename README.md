# DIANNE2.0
Codes and data in Socas-Navarro &amp; Asensio Ramos (2021)

Data for DKIST/ViSP will be posted here after the first observations become available

The training set for Hinode data may be downloaded from 
https://owncloud.iac.es/index.php/s/DZCdaliyILaK21A
and
https://owncloud.iac.es/index.php/s/RGTiLn8DQPkJ8W5
(profiles and parameters, respectively). This training set is created with the IDL procedure in create_database and the NICOLE code (see below) 


create_database

The IDL procedure prepare_syn.pro produces 1e6 random model atmospheres, creates a NICOLE input file with these models, calls NICOLE and reads the output spectral profiles. The code NICOLE is included in this directory as well


ANN

This directory contains the ANN model (ann.pth) and the Python code used to train it (train.py). The training code requires the files database.prof.idl and params.idl. These files are generated with the procedure in create_database (see above)


invert

This directory contains the IDL procedure readfiles.pro, which reads the Hinode observations in FITS format, and transforms the data to the format expected by the Python code that runs the ANN to invert the data (forward.py)

