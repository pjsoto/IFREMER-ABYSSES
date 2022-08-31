import os
import sys


Schedule = []


#Schedule.append("python Main_Script_For_Running_Shells_White_Fragments_Comitee_MS.py --cross_domain False")
#Schedule.append("python Main_Script_For_Running_Shells_White_Fragments_Comitee_WC.py --cross_domain False")
#Schedule.append("python Main_Script_For_Running_Shells_White_Fragments_Comitee_ET.py --cross_domain False")

Schedule.append("python Main_Script_For_Running_Lithology_Comitee_MS.py --cross_domain False")
Schedule.append("python Main_Script_For_Running_Lithology_Comitee_WC.py --cross_domain False")
Schedule.append("python Main_Script_For_Running_Lithology_Comitee_ET.py --cross_domain False")

#Schedule.append("python Main_Script_For_Running_Morphology_Comitee_MS.py --cross_domain False")
#Schedule.append("python Main_Script_For_Running_Morphology_Comitee_WC.py --cross_domain False")
#Schedule.append("python Main_Script_For_Running_Morphology_Comitee_ET.py --cross_domain False")

for i in range(len(Schedule)):
    os.system(Schedule[i])
