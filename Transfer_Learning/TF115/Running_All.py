import os
import sys


Schedule = []


Schedule.append("python Main_Script_For_Running_Shells_White_Fragments_MS.py --cross_domain True")
Schedule.append("python Main_Script_For_Running_Shells_White_Fragments_WC.py --cross_domain True")
Schedule.append("python Main_Script_For_Running_Shells_White_Fragments_ET.py --cross_domain True")

Schedule.append("python Main_Script_For_Running_Lithology_MS.py --cross_domain True")
Schedule.append("python Main_Script_For_Running_Lithology_WC.py --cross_domain True")
Schedule.append("python Main_Script_For_Running_Lithology_ET.py --cross_domain True")

Schedule.append("python Main_Script_For_Running_Morphology_MS.py --cross_domain True")
Schedule.append("python Main_Script_For_Running_Morphology_WC.py --cross_domain True")
Schedule.append("python Main_Script_For_Running_Morphology_ET.py --cross_domain True")

for i in range(len(Schedule)):
    os.system(Schedule[i])
