Human activity recognition through recurrent neural networks for human-robot interaction in agriculture

Athanasios Anagnostis(1,2,3), Lefteris Benos(1), Dimitrios Tsaopoulos(1) and Dionysis Bochtis(1,2)
1 · Institute for Bio-economy and Agri-technology, Center for Research and Technology Hellas, Thessaloníki, Greece 
2 · farmB Digital Agriculture, Doiraniis 17, Thessaloniki, GR54639, Greece
3 · Computer Science & Telecommunications Department, University of Thessaly, 35131, Lamia, Greece

#################################################################################################

The study involved 20 participants (13 male, 7 female), whose average age, height and weight were 30.95 years (SD 4.85), 1.75 m (SD 0.08) and 75.40 kg (SD 17.20), respectively.

Each participant had to sequentially perform a specific task which can be divided in six sub-activities, namely:
1.	Standing still until the signal is given to start;
2.	Walking a distance of 3.5 m without carrying any crate;
3.	Bending down to approach the crate;
4.	Lifting the crate from the ground to an upright position;
5.	Walking back the distance of 3.5 m with carrying the crate;
6.	Placing the crate onto the robot.

Two robots were utilized (Husky and Thorvald), which are usually used in outdoor environments. The two available robots correspond to a deposit height of the crate equal to 40 cm (Husky) and 80 cm (Thorvald). 
Furthermore, the crate was either empty (tare weight equal to 1.5 kg) or full with weight plates with a total mass (crate and plates) approximately equal to 20% of each participant’s mass. The mass of the available weight plates was 1 and 2.5 kg for the purpose of easily adjusting the required mass to be lifted and carried. 
An open plastic crate, commonly used in agriculture, was used with handles on both sides at 28 cm height above its base. The dimensions of the crate were 31 Χ 53 Χ 35 cm (height Χ width Χ depth). 
Consequently, each participant carried out four sub-cases:     
•	Empty crate – Husky;
•	Crate full with the required weight – Husky;
•	Empty crate – Thorvald;
•	Crate full with the required weight – Thorvald.

Each sub-case was performed 3 times in a randomised order and at each participant’s own pace. 

5 sensors (VICON IMeasureU Blue Trident) were attached to the chest (breast bone), cervix (approximately T1 vertebra), lumbar region (approximately L4), right and left wrist. 
They were attached via special velcro straps at the two wrists (provided by the manufacturer), while the rest three sensors were attached via double-sided tape. Each IMU encompasses a tri-axial accelerometer, a tri-axial gyroscope and a tri-axial magnetometer. 
The sampling frequency, that was used throughout the experimental sessions, was 50 Hz.
The Capture.U software (provided by VICON) was used to synchronise the sensors and capture the data, while the latter were saved directly to the sensors for further processing. Since Capture.U is available only for iOS devices, an Apple iPad mini (64 GB) was utilised for the present investigation.

Despite sensor temporal syncronisation, the raw sensor signals (accelerometer, gyroscope, magnetometer) were not pre-processed whatsoever.

The dataset is provided in a set of CSV files. Each file represents only one run of the defined activity, under  different conditions.The IMU sensor data includes the signals from the accelerometer, gyroscope and magnetometer (x-, y-, z-axis) of all subjects, absolute and relevant time (sec), and the labels of all the performed sub-activities labelled as "condition".

#################################################################################################

Auxiliary files are provided for further description of the sub-activity mapping and the experiments' encoding which is related to the CSV files' naming, as well as the licence under which this dataset can be used.
These are:

· 'README.txt': the present document.

· 'Raw_data/PTTT_SSSS.csv': The raw data including triaxial acceleration, angular velocity and direction signals for all subjects, identified by the character P, all experimental variations identified with numbers TTT and all sensors identified with the numbers SSSS. Each row contains the recorded time (sec), the absolute time (sec), the triaxial signals for acceleration, angular velocity and direction, as well as the class of the activity names as "condition".

· 'sub-activity_mapping.txt': The mapping of activities to the "condition" variable.

· 'file_naming.txt': The variations for each experiment conducted.

· 'LICENCE.GPL': file containing the disclaimer for the licence used for this dataset.

#################################################################################################

Additional notes:

· Each file has different length because the activities were conducted "at own pace".
· Data are not normalized.
· Acceleration is measured in m/s^2.
· Angular velocity is measured in rad/s.
· Direction is measured in μΤ.
· Recording of the experiments is not provided but can be requested at https://ibo.certh.gr/contact/.

#################################################################################################

License:

This dataset is under the GNU GENERAL PUBLIC LICENSE - Version 2 (please consult LICENCE.GPL file)
Research and publications with the use of this dataset must be acknowledged by the following citation:

· Human activity recognition through recurrent neural networks for human-robot interaction in agriculture, Anagnostis A., Benos E., Tsaopoulos D., Bochtis D., Appl. Sci. 2021, 11, 2188. https://doi.org/10.3390/app11052188

#################################################################################################

Acknowledgements:

Athanasios Anagnostis, Lefteris Benos, Dimitrios Tsaopoulos and Dionysis Bochtis.
Institute for Bio-economy and Agri-technology, Center for Research and Technology Hellas
January 2021
