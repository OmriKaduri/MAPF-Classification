import os
import glob
import signal
import subprocess
from subprocess import Popen, PIPE, TimeoutExpired
from time import monotonic as timer
from subprocess import check_output
import csv
# picat_file = 'picats/' + '-'.join([str(x) for x in [map_name, num_agents, instance_id]]) + '.pi'
allowed_maps = ['empty-8-8', 'empty-16-16']

base_picat_path = os.path.join('..', '..', 'MAPF', 'classification', 'graph', 'base-picats/')
# base_picat_path = 'base-picats/'
picats_path = os.path.join('..', '..', 'MAPF', 'classification', 'graph', 'picats')
# picats_path = 'picats/*'
if __name__ == "__main__":
    results_file = 'results.csv'
    with open(results_file,'w+') as results:
        # print(os.getcwd(), base_picat_path)
        print(picats_path)

        for base_picat in glob.glob(base_picat_path+'*'):
            allow = False
            for allowed in allowed_maps:
                if allowed in base_picat:
                    allow = True
            if not allow:
                continue
            print(base_picat)
            picat_name = base_picat.split('-base')[0].split('/')[-1]
            print(picat_name)
            with open(base_picat, 'r') as base:
                lines = []
                for line in base:
                    lines.append(line)

                for picat_file in glob.glob(f'{picats_path}/{picat_name}*'):
                    picat__file_name = picat_file.split('/')[-1]
                    print(picat__file_name )
                    tmp_file = 'tmp.pi'
                    with open(tmp_file,'w+') as tmp_pi, open(picat_file,'r') as pi_instance:
                        for line in lines:
                            tmp_pi.write(line)
                        tmp_pi.write('\n')
                        for line in pi_instance:
                            tmp_pi.write(line)
                    start = timer()
                    try:
                        check_output(['./picat', 'soc.pi',f'{tmp_file}'], timeout=300)
                        results.write(f'{picat__file_name},{timer() - start},1\n')
                        
                    except TimeoutExpired :
                        results.write(f'{picat__file_name},{timer() - start},0\n')
                        print("Expired")
                    except subprocess.CalledProcessError:
                        results.write(f'{picat__file_name},{timer() - start},0\n')
                        print("CalledError")
                    # with Popen('sleep 30 ', shell=True, stdout=PIPE) as process:
                    #     try:
                    #         output = process.communicate(timeout=1)[0]
                    #     except TimeoutExpired:
                    #         os.killpg(process.pid, signal.SIGINT)  # send signal to the process group
                    #         output = process.communicate()[0]
                    print('Elapsed seconds: {:.2f}'.format(timer() - start))
