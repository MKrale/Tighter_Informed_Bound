#!/bin/bash

#       ...
nmbr_cores=100
# nmbr_cores=12

##########################################################
#        Snake Maze Environment:
##########################################################

echo -e "\n\n============= UPPER BOUND TESTS  =============\n\n"

folder_path="Data/UpperBounds/"

# for env in "ABC" "Tiger" "RockSample5" "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake4" "FrozenLake10" "Hallway1" "Hallway2" "MiniHallway" "TigerGrid" "Tag" # ALL
 for env in "ABC" "RockSample5" "FrozenLake4" "Tiger" # QUICK
# for env in "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake10" "Tag" "SparseHallway1" "SparseHallway2" "SparseTigerGrid"  # LONG
do
   julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.95 &
   julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.99 &
done
wait

echo -e "\n\n============= SARSOP TESTS  =============\n\n"

folder_path="Data/SarsopTest/"
i=0

# for env in "ABC" "Tiger" "RockSample5" "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake4" "FrozenLake10" "Hallway1" "Hallway2" "MiniHallway" "TigerGrid" "Tag" # ALL
 for env in "ABC" "RockSample5" "FrozenLake4" "Tiger" # QUICK
# for env in "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake10" "Tag" "SparseHallway1" "SparseHallway2" "SparseTigerGrid" # LONG
# for env in "Hallway1" "Hallway2" "MiniHallway" "TigerGrid"
# for env in "K-out-of-N3" "FrozenLake10"
do
    julia --project=. run_sarsoptest.jl --env $env --timeout 3200.0 --path $folder_path --discount 0.95 &
    julia --project=. run_sarsoptest.jl --env $env --timeout 3200.0 --path $folder_path --discount 0.99 &
done
#wait

echo -e "\n\n============= RUNS COMPLETED =============\n\n"

