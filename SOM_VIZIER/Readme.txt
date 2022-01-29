AA/2021/40766
================================================================================
Machine learning methods for constructing probabilistic Fermi-LAT catalogs 
   			Bhat A., Malyshev D.
      
================================================================================
ADC_Keywords: Astrophysics - High Energy Astrophysical Phenomena; Astrophysics - 
	      Instrumentation and Methods for Astrophysics
Keywords: Methods:statistical, Catalogs, Gamma rays: general

Abstract:
Context: Classification of sources is one of the most important tasks in 
astronomy. Sources detected in one wavelength band, for example using gamma rays, 
may have several possible associations in other wavebands, or there may be no 
plausible association candidates.
Aims: In this work we aim to determine the probabilistic classification of 
unassociated sources in the third Fermi Large Area Telescope (LAT) point source 
catalog (3FGL) and the fourth Fermi LAT data release 2 point source catalog 
(4FGL-DR2) using two classes – pulsars and active galactic nuclei (AGNs) – or 
three classes – pulsars, AGNs, and “OTHER” sources.
Methods: We use several machine learning (ML) methods to determine a probabilistic
classification of Fermi-LAT sources.We evaluate the dependence of results on the 
meta parameters of the ML methods, such as the maximal depth of the trees in 
tree-based classification methods and the number of neurons in neural networks.
Results: We determine a probabilistic classification of both associated and 
unassociated sources in the 3FGL and 4FGL-DR2 catalogs. We cross-check the 
accuracy by comparing the predicted classes of unassociated sources in 3FGL with
their associations in 4FGL-DR2 for cases where such associations exist. We find 
that in the two-class case it is important to correct for the presence of OTHER
sources among the unassociated ones in order to realistically estimate the 
number of pulsars and AGNs.We find that the three-class classification, despite
different types of sources in the OTHER class, has a similar performance as the
two-class classification in terms of reliability diagrams and, at the same 
time, it does not require adjustment due to presence of the OTHER sources among
the unassociated sources. We show an example of the use of the probabilistic 
catalogs for population studies, which include associated and unassociated 
sources.

Description:
The ten files describe 4+4 probabilistic catalogs plus 2 best candidate lists. 
These are 3FGL and 4FGL-DR2 based 2 and 3 class catalogs (4),there respective 
SMOTE counterparts, and PSR and OTHER sources which are unassociated but have a
high probabilitiy of being a PSR or an OTHER source.

File Summary:
--------------------------------------------------------------------------------
 FileName     			     Columns    Rows    Caption
--------------------------------------------------------------------------------
ReadMe            			        127     This file
3FGL_4FGL-DR2_Candidates_PSR            29      112   	PSR candidates using both
							catalogs
3FGL_prob_catalog_2classes		52	3034	2-class classification
3FGL_prob_catalog_2classes_SMOTE	33	3034	2-class using SMOTE
3FGL_prob_catalog_3classes		65	3034	3-class classification
3FGL_prob_catalog_3classes_SMOTE	40	3034	3-class using SMOTE
4FGL-DR2_Candidates_OTHER_3classes	30	15	OTHER candidates using 
							4FGL-DR2
4FGL-DR2_prob_catalog_2classes		56	5788	2-class for 4FGl-DR2
4FGL-DR2_prob_catalog_2classes_SMOTE	39	5788	2-class using SMOTE
4FGL-DR2_prob_catalog_3classes		72	5788	3-class for 4FGL-DR2
4FGL-DR2_prob_catalog_3classes_SMOTE    47      5788    3-class using SMOTE            
--------------------------------------------------------------------------------

Column nomenclature used in the files
--------------------------------------------------------------------------------
Name	    		Type    Explanations
--------------------------------------------------------------------------------
Source_Name  		String  Source Name in the Fermi LAT catalog
GLON	     		Double	Galactic Longitude
GLAT	     		Double  Galactic Latitude
ln(Energy_Flux100) 	Double	log of Energy_Flux100
ln(Unc_Energy_Flux100)  Double	log of uncertainty on the above
ln(Pivot_Energy)	Double	log of Pivot Energy
LP_Index		Double
Unc_LP_Index		Double
LP_beta			Double
LP_SigCurv		Double
ln(Variability_Index)   Double  log of variablity index
HR_ij			Double	Hardness ratios in 5 and 7 bands for 3FGL
				and 4FGL-DR2 respectively. i,j correspond
				to the band numbers
ln(Signif_Curve)	Double	log of signif_curve
500MeV_Index		Double	Index at 500 MeV for 3FGL catalog
Pivot_Energy		Double
Spectrum_Type		String	Type of spectral fit		  
Category_		String	Category based on association in 3FGL or 4FGl-DR2
Class_			String	Class based on association in 3FGL or 4FGL_DR2
ASSOC_FGL		String	Past associations in Fermi LAT
ASSOC_FHL		String	Past associations in FHL
AGN_x_y			Double	Probability of being an AGN with x being the machine 
				learning method and y being the catalog. x 
				corresponds to BDT, NN, RF, and LR, while y is 4FGL
				or 3FGL
AGN_x_STD_y		Double	Corresponding standard deviation
PSR_x_y			Double	Probability of being a PSR
PSR_x_STD_y		Double	Corresponding standard deviation for PSR prob
OTHER_x_y		Double	Probability of being an OTHER source for 3-class 
				classification
OTHER_x_STD_y		Double	Corresponding standard deviation for OTHER prob
AGN_x_O_y		Double	Probability of being an AGN with x being the machine 
				learning method and y being the catalog for normal
				oversampling
AGN_x_STD_O_y		Double	Corresponding standard deviation 
PSR_x_O_y		Double	Probability of being a PSR for normal oversampling
PSR_x_STD_O_y		Double	Corresponding standard deviation 
OTHER_x_O_y		Double	Probability of being an OTHER source for 3-class 
				classification for normal oversampling
OTHER_x_STD_O_y		Double	Corresponding standard deviation for OTHER prob
Flags_			Short	Flags in 4FGL_DR2 or 3FGL catalogs
Missing_values_flag	Double	0 for no missing values in features and 1 for
				a missing feature value
Category_Prob_		Double	Describes the category based on the highest 
				probabilities and using all methods:
				 AGN, PSR, OTHER, MIXED.
PSR_TOTAL		Double	Sum of PSR probabilities of all methods 
AGN_TOTAL		Double	Sum of AGN probabilities of all methods 
OTHER_TOTAL		Double	Sum of OTHER probabilities of all methods
--------------------------------------------------------------------------------
Note (1): For all columns an _4FGL or _3FGL is present to make it clear which 
catalog was used for it, or in the case of features, which catalog they were taken
from.
Note (2): Column names for the SMOTE oversampling are also the same but with a _S
instead of _O attached.
Note (3): PSR and OTHER candidate lists have Separation column which gives the 
separation to an object in arkseconds. 
--------------------------------------------------------------------------------
================================================================================
(End)                           
