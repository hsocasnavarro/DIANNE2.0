# Paper_SNAR21
Codes and data in Socas-Navarro &amp; Asensio Ramos (2021)

Data for DKIST/ViSP will be posted here after the first observations become available

The training set for Hinode data may be downloaded from 
https://owncloud.iac.es/index.php/s/DZCdaliyILaK21A
and
https://owncloud.iac.es/index.php/s/RGTiLn8DQPkJ8W5
(profiles and parameters, respectively). This training set is created with the IDL procedure in create_database and the NICOLE code (see below) 

create_database

The IDL procedure prepare_syn.pro creates 1e6 model atmospheres, creates a NICOLE input file with these models, calls NICOLE and reads the output spectral profiles. 
