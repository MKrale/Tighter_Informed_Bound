#!/bin/bash

#       This file runs all experiments as used in the paper, and stores them in the data-folder. For creating plots, see the Plot_Data.ipynb
#       Note: this file has been written to efficiently use all resources on our setup. 
#       Other setups may require altering the code such that less (or more) runs occur at once.
#       Furthermore, in running all code in this file may take a long time: in practice we ran our experiments in stages.

nmbr_cores=100
# nmbr_cores=12

##########################################################
#        Snake Maze Environment:
##########################################################

echo -e "\n\n============= Snake Maze, changing alpha  =============\n\n"

folder_path="Data/"
i=0

# for env in "ABC" "Tiger" "RockSample5" "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake4" "FrozenLake10" "Tag" # ALL
# for env in "ABC" "RockSample5" "FrozenLake4" #"Tiger" # QUICK
# for env in "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake10" "Tag" # LONG
for env in "Hallway1" "Hallway2" "MiniHallway" "TigerGrid"
do
    julia --project=. run_sarsoptest.jl --env $env --timeout 1800 --path $folder_path &
done

wait
echo -e "\n\n============= RUNS COMPLETED =============\n\n"

