import os

config_files = os.listdir('./evolver/configs')

while True:
    for config in config_files:
        os.system('cp %s ./evolver/race_config/quickrace.xml'%('./evolver/configs/' + config))

        # os.system('./start.sh -p 3001')
        os.system('torcs -t 100000 -nofuel -nodamage -nolaptime -r /home/student/Documents/torcs-server/torcs-client/evolver/race_config/quickrace.xml')
