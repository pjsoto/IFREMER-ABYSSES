import os
import sys
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--running_in', dest='running_in', type=str, default='Datarmor_PBS', help='Decide wether the script will be running')
args = parser.parse_args()

if args.running_in == 'Datarmor_Interactive':
    TestComitee_MAIN_COMMAND = ""
if args.running_in == 'Datarmor_PBS':
    TestComitee_MAIN_COMMAND = "$HOME/CODE/IFREMER-ABYSSES/Transfer_Learning/TF115/"

Schedule = []


#Schedule.append("python Main_Script_For_Running_Shells_White_Fragments_Comitee_MS.py --cross_domain False")
#Schedule.append("python Main_Script_For_Running_Shells_White_Fragments_Comitee_WC.py --cross_domain False")
#Schedule.append("python Main_Script_For_Running_Shells_White_Fragments_Comitee_ET.py --cross_domain False")

#Schedule.append("python Main_Script_For_Running_Lithology_Comitee_MS.py --cross_domain False")
#Schedule.append("python Main_Script_For_Running_Lithology_Comitee_WC.py --cross_domain False")
#Schedule.append("python Main_Script_For_Running_Lithology_Comitee_ET.py --cross_domain False")

Schedule.append("python " + TestComitee_MAIN_COMMAND + " Main_Script_For_Running_Morphology_Comitee_MS.py --cross_domain False")
Schedule.append("python " + TestComitee_MAIN_COMMAND + " Main_Script_For_Running_Morphology_Comitee_WC.py --cross_domain False")
Schedule.append("python " + TestComitee_MAIN_COMMAND + " Main_Script_For_Running_Morphology_Comitee_ET.py --cross_domain False")

for i in range(len(Schedule)):
    os.system(Schedule[i])
