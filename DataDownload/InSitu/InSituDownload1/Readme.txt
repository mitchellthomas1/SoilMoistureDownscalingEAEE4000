Variables stored in separate files (Header+values)

Filename

	Data_separate_files_header_startdate(YYYYMMDD)_enddate(YYYYMMDD)_userid_randomstring_currrentdate(YYYYMMDD).zip
	
	e.g., Data_separate_files_header_20050316_20050601.zip

	
Folder structure

	Networkname
		Stationname

		
Dataset Filename

	CSE_Network_Station_Variablename_depthfrom_depthto_startdate_enddate.ext

	CSE	- Continental Scale Experiment (CSE) acronym, if not applicable use Networkname
	Network	- Network abbreviation (e.g., OZNET)
	Station	- Station name (e.g., Widgiewa)
	Variablename - Name of the variable in the file (e.g., Soil-Moisture)
	depthfrom - Depth in the ground in which the variable was observed (upper boundary)
	depthto	- Depth in the ground in which the variable was observed (lower boundary)
	startdate -	Date of the first dataset in the file (format YYYYMMDD)
	enddate	- Date of the last dataset in the file (format YYYYMMDD)
	ext	- Extension .stm (Soil Temperature and Soil Moisture Data Set see CEOP standard)
	
	e.g., OZNET_OZNET_Widgiewa_Soil-Temperature_0.150000_0.150000_20010103_20090812.stm

	
File Content Sample
	
	REMEDHUS   REMEDHUS        Zamarron          41.24100    -5.54300  855.00    0.05    0.05  (Header)
	2005/03/16 00:00    10.30 U	M	(Records)
	2005/03/16 01:00     9.80 U M

	
Header

	CSE Identifier - Continental Scale Experiment (CSE) acronym, if not applicable use Networkname
	Network	- Network abbreviation (e.g., OZNET)
	Station	- Station name (e.g., Widgiewa)
	Latitude - Decimal degrees. South is negative.
	Longitude - Decimal degrees. West is negative.
	Elevation - Meters above sea level
	Depth from - Depth in the ground in which the variable was observed (upper boundary)
	Depth to - Depth in the ground in which the variable was observed (lower boundary)

	
Record

	UTC Actual Date and Time
	yyyy/mm/dd HH:MM
	Variable Value
	ISMN Quality Flag
	Data Provider Quality Flag, if existing


Network Information

	ARM
		Abstract: The soil moisture datasets collected at ARM facilities originates from two different instruments. SWATS instrument measure soil moisture in two different profiles at 8 depths from 0,05 to 1,75m, the SEBS instrument measure three profiles in depth of 0,025m. The site is managed by U.S. Department of Energy as part of the Atmospheric Radiation Measurement Climate Research Facility.
		Continent: Americas
		Country: USA
		Stations: 35
		Status: running
		Data Range: from 1996-02-05 
		Type: project
		Url: http://www.arm.gov/
		Reference: Cook, D. R. (2016a), Soil temperature and moisture proﬁle (stamp) system handbook, Technical report, DOE Oﬃce of Science Atmospheric Radiation Measurement (ARM) Program. https://www.osti.gov/biblio/1332724;

Cook, D. R. (2016b), Soil water and temperature system (swats) instrument handbook, Technical report, DOE Oﬃce of Science Atmospheric Radiation Measurement (ARM) Program. https://www.osti.gov/biblio/1004944;

Cook, D. R. & Sullivan, R. C. (2018), Surface energy balance system (sebs) instrument handbook, Technical report, DOE Office of Science Atmospheric Radiation Measurement (ARM) Program. https://www.arm.gov/publications/tech_reports/handbooks/sebs_handbook.pdf;
		Variables: precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.02 - 0.02 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.15 - 0.15 m, 0.20 - 0.20 m, 0.25 - 0.25 m, 0.35 - 0.35 m, 0.50 - 0.50 m, 0.60 - 0.60 m, 0.75 - 0.75 m, 0.80 - 0.80 m, 0.85 - 0.85 m, 1.25 - 1.25 m, 1.75 - 1.75 m
		Soil Moisture Sensors: Hydraprobe II Sdi-12 E, Hydraprobe II Sdi-12 S, Hydraprobe II Sdi-12 W, SMP1, Water Matric Potential Sensor 229L, 

	AWDN
		Abstract: In 1998 the High Plains Regional Climate Center upgraded stations in the Automated Weather Data Network (AWDN) to monitor soil water at 14 sites in Nebraska. The Natural Resources Conservation Service (NRCS) has provided soil moisture sensors for some of the sites. Currently, sensors are placed at 10, 25, 50, and 100 cm below the surface at each site. AWDN technicians maintain the sensors and laboratory facilities were updated to include adequate test and calibration equipment. The network archives the data for a daily time step.
		Continent: Americas
		Country: USA
		Stations: 50
		Status: running
		Data Range: from 1998-01-01  to 2010-12-30
		Type: project
		Url: https://hprcc.unl.edu/awdn/
		Reference: We acknowledge the work of Natalie Umphlett and the AWDN team in support of the ISMN;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.10 - 0.10 m, 0.25 - 0.25 m, 0.50 - 0.50 m, 1.00 - 1.00 m
		Soil Moisture Sensors: ThetaProbe ML2X, Vitel, 

	BNZ-LTER
		Abstract: The Bonanza Creek (BNZ) LTER project was initiated in 1987 and since then has provided experimental and observational research designed to understand the dynamics, resilience, and vulnerability of Alaska"s boreal forest ecosystems. The project has illuminated the responses of boreal forest organisms and ecosystems to climate and various atmospheric inputs, focusing on forest and landscape dynamics and biogeochemistry. This project will continue that long-term line of research, expanding it to broaden the landscape under study, broaden the predictive realm of the resulting information, and to directly address the resilience of socio-economic systems. The project hypothesizes that the past observed high resilience of boreal ecosystems to interannual and decadal changes in environmental conditions is approaching a critical tipping point., 
This project contributes to understanding of the structure, function, and dynamics of boreal forest ecosystems and the broader boreal landscape, including the human communities. It assembles and integrates valuable long-term data sets on climate, hydrology, biology, ecology, and biogeochemical and geomorphic processes, as incorporates emerging data types, including molecular and social science data and digital images. The project has broad societal value through its contributions to knowledge that can inform management of boreal forest ecosystems and sustainability of subsistence communities. Its broader values also include extensive research-based training and educational program development. Its strong public outreach program includes collaborations between artists and scientists and strong linkages with Native organizations.
		Continent: Americas
		Country: Alaska
		Stations: 12
		Status: running
		Data Range: from 1989-05-01  to 2012-12-31
		Type: project
		Url: http://www.lter.uaf.edu/
		Reference: Van Cleve, Keith, Chapin, F.S. Stuart, Ruess, Roger W. 2015. Bonanza Creek Long Term Ecological Research Project Climate Database - University of Alaska Fairbanks.,
Bonanza Creek Long Term Ecological Research Project Climate Database. 2015. University of Alaska Fairbanks. https://www.lter.uaf.edu/;
		Variables: air temperature, precipitation, snow depth, snow water equivalent, soil moisture, soil temperature, surface temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.50 - 0.50 m
		Soil Moisture Sensors: CS615, 

	FLUXNET-AMERIFLUX
		Abstract: Datasets of 2 stations near the city of Sacramento are provided. They are managed by Dennis D. Baldocchi, University of California, Berkeley. 
		Continent: Americas
		Country: USA
		Stations: 8
		Status: running
		Data Range: from 2000-10-22 
		Type: project
		Url: http://ameriflux.lbl.gov/
		Reference: We acknowledge the work of Dennis D. Baldocchi and the FLUXNET-AMERIFLUX team in support of the ISMN;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.00 m, 0.00 - 0.15 m, 0.02 - 0.02 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.15 - 0.30 m, 0.20 - 0.20 m, 0.30 - 0.45 m, 0.45 - 0.60 m, 0.50 - 0.50 m
		Soil Moisture Sensors: CS655, Moisture Point PRB-K, ThetaProbe ML2X, ThetaProbe ML3, 

	ICN
		Abstract: "Soil moisture datasets for 19 stations, collected and processed by the "Illinois State Water Survey". The soil moisture data given in total soil moisture [mm] for 11 layers of each station were converted to volumetric soil moisture [m^3/m^3]
"
		Continent: Americas
		Country: USA
		Stations: 19
		Status: inactive
		Data Range: from 1981-01-01  to 2010-11-21
		Type: project
		Url: http://www.isws.illinois.edu/warm/
		Reference: Hollinger, Steven E., and Scott A. Isard, 1994: A soil moisture climatology of Illinois. J. Climate, 7, 822-833., https://doi.org/10.1175/1520-0442(1994)007<0822:ASMCOI>2.0.CO,2;
		Variables: precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.10 m, 0.10 - 0.30 m, 0.30 - 0.50 m, 0.50 - 0.70 m, 0.70 - 0.90 m, 0.90 - 1.10 m, 1.10 - 1.30 m, 1.30 - 1.50 m, 1.50 - 1.70 m, 1.70 - 1.90 m, 1.90 - 2.00 m
		Soil Moisture Sensors: Stevens Hydra Probe, Troxler Neutron Depth Probe, Troxler Neutron Surface Probe, 

	IOWA
		Abstract: Soil moisture measurements for two different catchments with three sites per catchment were taken in the southwestern part of Iowa. Soil moisture observations during the period 1972 to 1994 are available. Measurements were taken at 12 different soil layers up to a depth of 2.6 m. The gravimetric method was used for the top layers and a neutron probe for the lower layers. On average observations were made twice a month between April and October.
		Continent: Americas
		Country: USA
		Stations: 6
		Status: inactive
		Data Range: from 1972-04-04  to 1994-11-15
		Type: project
		Url: 
		Reference: Robock, Alan, Konstantin Y. Vinnikov, Govindarajalu Srinivasan, Jared K. Entin, Steven E. Hollinger, Nina A. Speranskaya, Suxia Liu, and A. Namkhai, 2000: The Global Soil Moisture Data Bank. Bull. Amer. Meteorol. Soc., 81, 1281-1299, https://doi.org/10.1175/1520-0477(2000)081<1281:TGSMDB>2.3.CO,2;
		Variables: soil moisture, 
		Soil Moisture Depths: 0.00 - 0.08 m, 0.08 - 0.15 m, 0.15 - 0.30 m, 0.30 - 0.46 m, 0.46 - 0.69 m, 0.69 - 0.84 m, 0.84 - 1.07 m, 1.07 - 1.37 m, 1.37 - 1.68 m, 1.68 - 1.98 m, 1.98 - 2.29 m, 2.29 - 2.59 m
		Soil Moisture Sensors: n.s., 

	iRON
		Abstract: 
		Continent: Americas
		Country: USA
		Stations: 10
		Status: running
		Data Range: from 2012-06-07 
		Type: meteo
		Url: https://agci.org/iron
		Reference: Osenga, E.C., Vano, J.A., and Arnott, J.C. 2021. A community-supported weather and soil moisture monitoring database of the Roaring Fork catchment of the Colorado River Headwaters. Hydrologic Processes, 2021:35. https://doi.org/10.1002/hyp.14081;

Osenga, E. C., Arnott, J. C., Endsley, K. A., & Katzenberger, J. W. (2019). Bioclimatic and soil moisture monitoring across elevation in a mountain watershed: Opportunities for research and resource management. Water Resources Research, 55. https://doi.org/10.1029/2018WR023653;
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.20 - 0.20 m, 0.50 - 0.50 m, 0.51 - 0.51 m, 1.00 - 1.00 m
		Soil Moisture Sensors: 10HS, EC5, EC5 I, EC5 II, HMP155, Hydraprobe II, 

	LAB-net
		Abstract: In Chile, remote sensing applications related to soil moisture and evapotranspiration estimates have increased during the last decades because of the drought and the water use conflicts which generate a strong interest on water demand. To address these problems on water balance in large scales by using remote sensing imagery, LAB-net was created as the first soil moisture network located in Chile over four different land cover types. These land cover types are vineyard and olive orchards located in a semi-arid region of Copiapó Valley, a well irrigated raspberry crop located in the central zone of of Tinguiririca Valley and a green grass rangeland located Austral zone of Chile. Over each site, a well implemented meteorological station is continuously measuring a 5 minute intervals above the following parameters: soil moisture and temperature at two ground levels (5 and 20 cm), air temperature and relative humidity, net radiation, global radiation, brightness surface temperature (8 – 14 µm), rainfall and ground fluxes. This is the first approach of an integrated soil moisture network in Chile. The data generated by this network is freely available for any research or scientific purpose related to current and future soil moisture satellite missions.    

		Continent: Americas
		Country: Chile
		Stations: 4
		Status: running
		Data Range: from 2014-07-18 
		Type: meteo
		Url: http://www.biosfera.uchile.cl/LAB-net.html

		Reference: Mattar, C., Santamaria-Artigas, A., Duran-Alarcon, C., Olivera-Guerra, L., Fuster, R. & Borvar´an, D. (2016), ‘The lab-net soil moisture network: Application to thermal remote sensing and surface energy balance’, Data 1(1), https://doi.org/10.3390/data1010006;

Mattar, C., Santamaría-Artigas, A., Durán-Alarcón, C., Olivera-Guerra, L. and Fuster, R., 2014, September. LAB-net the first Chilean soil moisture network for remote sensing applications. In Quantitative Remote Sensing Symposium (RAQRS) (pp. 22-26);
		Variables: air temperature, precipitation, soil moisture, soil temperature, 
		Soil Moisture Depths: 0.07 - 0.07 m, 0.10 - 0.10 m, 0.20 - 0.20 m
		Soil Moisture Sensors: CS616, CS650, CS655, 

	PBO_H2O
		Abstract: The soil moisture data is measured using GPS reflections.
		Continent: Americas
		Country: USA
		Stations: 163
		Status: inactive
		Data Range: from 2007-01-01 
		Type: project
		Url: https://gnss-reflections.org/maps?product=smc
		Reference: Kristine M. Larson, Eric E. Small, Ethan D. Gutmann, Andria L. Bilich, John J. Braun, Valery U. Zavorotny: Use of GPS receivers as a soil moisture network for water cycle studies. GEOPHYSICAL RESEARCH LETTERS, VOL. 35, L24405, https://doi.org/10.1029/2008GL036013, 2008;
		Variables: precipitation, air temperature, soil moisture, snow depth, 
		Soil Moisture Depths: 0.00 - 0.05 m
		Soil Moisture Sensors: GPS, 

	RISMA
		Abstract: In 2010 and 2011, Agriculture and Agri-Food Canada (AAFC), with the collaboration of Environment Canada, established three in situ monitoring networks near Kenaston (Saskatchewan), Carman (Manitoba) and Casselman (Ontario) as part of the Sustainable Agriculture Environmental Systems (SAGES) project titled Earth Observation Information on Crops and Soils for Agri-Environmental Monitoring in Canada.
		Continent: Americas
		Country: Canada
		Stations: 24
		Status: running
		Data Range: from 2013-06-15 
		Type: project
		Url: http://aafc.fieldvision.ca/
		Reference: Ojo, E. R., Bullock, P. R., L’Heureux, J., Powers, J., McNairn, H., & Pacheco, A. (2015). Calibration and evaluation of a frequency domain reflectometry sensor for real-time soil moisture monitoring. Vadose Zone Journal, 14(3), https://doi.org/10.2136/vzj2014.08.0114;

L’Heureux, J. (2011). 2011 Installation Report for AAFC‐ SAGES Soil Moisture Stations in Kenaston, SK. Agriculture;

Canisius, F. (2011). Calibration of Casselman, Ontario Soil Moisture Monitoring Network, Agriculture and Agri‐Food Canada, Ottawa, ON, 37pp;
		Variables: precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.00 - 0.05 m, 0.05 - 0.05 m, 0.20 - 0.20 m, 0.50 - 0.50 m, 1.00 - 1.00 m, 1.50 - 1.50 m
		Soil Moisture Sensors: Hydraprobe II Sdi-12, 

	SOILSCAPE
		Abstract: 
		Continent: Americas
		Country: USA
		Stations: 171
		Status: running
Data Range: 

		Type: project
		Url: http://soilscape.usc.edu/
		Reference: Moghaddam, M., A.R. Silva, D. Clewley, R. Akbar, S.A. Hussaini, J. Whitcomb, R. Devarakonda, R. Shrestha, R.B. Cook, G. Prakash, S.K. Santhana Vannan, and A.G. Boyer. 2016. Soil Moisture Profiles and Temperature Data from SoilSCAPE Sites, USA. ORNL DAAC, Oak Ridge, Tennessee, USA. http://dx.doi.org/10.3334/ORNLDAAC/1339;

Moghaddam, M., D. Entekhabi, Y. Goykhman, K. Li, M. Liu, A. Mahajan, A. Nayyar, D. Shuman, and D. Teneketzis, A wireless soil moisture smart sensor web using physics-based optimal control: concept and initial demonstration IEEE-JSTARS, , vol. 3, no. 4, pp. 522-535, December 2010, https://doi.org/10.1109/JSTARS.2010.2052918;

Shuman, D. I., Nayyar, A., Mahajan, A., Goykhman, Y., Li, K., Liu, M., Teneketzis, D., Moghaddam, M. & Entekhabi, D. (2010), ‘Measurement scheduling for soil moisture sensing: From physical models to optimal control’, Proceedings of the IEEE 98(11), 1918–1933, https://doi.org/10.1109/JPROC.2010.2052532;
		Variables: soil temperature, soil moisture, 
		Soil Moisture Depths: 0.04 - 0.04 m, 0.05 - 0.05 m, 0.10 - 0.10 m, 0.13 - 0.13 m, 0.15 - 0.15 m, 0.17 - 0.17 m, 0.20 - 0.20 m, 0.23 - 0.23 m, 0.28 - 0.28 m, 0.29 - 0.29 m, 0.30 - 0.30 m, 0.32 - 0.32 m, 0.35 - 0.35 m, 0.36 - 0.36 m, 0.37 - 0.37 m, 0.38 - 0.38 m, 0.39 - 0.39 m, 0.40 - 0.40 m, 0.41 - 0.41 m, 0.42 - 0.42 m, 0.43 - 0.43 m, 0.45 - 0.45 m, 0.46 - 0.46 m, 0.47 - 0.47 m, 0.48 - 0.48 m, 0.50 - 0.50 m, 0.60 - 0.60 m, 0.90 - 0.90 m
		Soil Moisture Sensors: EC5, 

	TxSON
		Abstract: 
		Continent: Americas
		Country: USA
		Stations: 41
		Status: running
		Data Range: from 2014-10-01 
		Type: project
		Url: http://www.beg.utexas.edu/research/programs/txson
		Reference: Caldwell, T. G., T. Bongiovanni, M. H. Cosh, T. J. Jackson, A. Colliander, C. J. Abolt, R. Casteel, T. Larson, B. R. Scanlon, and M. H. Young (2019), The Texas Soil Observation Network: A comprehensive soil moisture dataset for remote sensing and land surface model validation, Vadose Zone Journal, 18:100034, doi:10.2136/vzj2019.04.0034
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.05 - 0.05 m
		Soil Moisture Sensors: CS655, 

	USCRN
		Abstract: Soil moisture NRT network USCRN (Climate Reference Network) in United States;the  datasets of 114 stations were collected and processed by the National Oceanicand Atmospheric Administration"s National Climatic Data Center (NOAA"s NCDC)
		Continent: Americas
		Country: USA
		Stations: 115
		Status: running
		Data Range: from 2009-06-09 
		Type: meteo
		Url: http://www.ncdc.noaa.gov/crn/
		Reference: Bell, J. E., M. A. Palecki, C. B. Baker, W. G. Collins, J. H. Lawrimore, R. D. Leeper, M. E. Hall, J. Kochendorfer, T. P. Meyers, T. Wilson, and H. J. Diamond. 2013: U.S. Climate Reference Network soil moisture and temperature observations. J. Hydrometeorol., 14, 977-988, https://doi.org/10.1175/JHM-D-12-0146.1;
		Variables: surface temperature, precipitation, soil temperature, air temperature, soil moisture, 
		Soil Moisture Depths: 0.05 - 0.05 m, 0.10 - 0.10 m, 0.20 - 0.20 m, 0.50 - 0.50 m, 1.00 - 1.00 m
		Soil Moisture Sensors: Stevens Hydraprobe II Sdi-12, 

	USDA-ARS
		Abstract: The USDA-ARS watershed network was initiated as part of an EOS Aqua AMSR-E project in 2002. It consists of four sub-networks - Little River, Little Washita, Walnut Gulch and Reynolds Creek. These sub-networks have 16 to 29 stations on the area of few hundred km square. The soil moisture and soil temperature in the layer 0.00-0.05m are measured at each of the stations and the arithmetical average and area weighted average of the stations measurements are provided for each of the sub-networks. 
		Continent: Americas
		Country: USA
		Stations: 4
		Status: inactive
		Data Range: from 2002-06-01  to 2009-07-31
		Type: project
		Url: https://www.ars.usda.gov/
		Reference: Jackson, T.J., Cosh, M.H., Bindlish, R., Starks, P.J., Bosch, D.D., Seyfried, M.S., Goodrich, D.C., Moran, M.S., Validation of Advanced Microwave Scanning Radiometer Soil Moisture Products. IEEE Transactions on Geoscience and Remote Sensing. 48: 4256-4272, 2010, https://doi.org/10.1109/TGRS.2010.2051035;
		Variables: soil moisture, soil temperature, 
		Soil Moisture Depths: 0.00 - 0.05 m
		Soil Moisture Sensors: Hydraprobe Analog (2.5 Volt) - area weighted average, Hydraprobe Analog (2.5 Volt) - average, 

