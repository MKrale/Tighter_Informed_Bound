#!/bin/bash

# ... description ...
folder_path="Data/"
echo -e "\n\n============= Manual Test  =============\n\n"
for env in "RockSample5" # QUICK
do
   julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.95 &
   # julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.99 &
done
wait

echo -e "\n\n============= RUNS COMPLETED =============\n\n"

##################


# 30 tasks, generally takes < 10 min to run

# Upper bounds:
#folder_path="Data/UpperBounds/"
#for env in "ABC" "RockSample5" "RockSample10" "FrozenLake4" "FrozenLake10" "Tiger" "K-out-of-N2" # QUICK
#do
    
#    julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.95 &
#    julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.99 &
#done

#for env in "ABC" "RockSample5" "FrozenLake4" "Tiger" # QUICK
#do
#    julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.95 --solver WBIB &
#    julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.99 --solver WBIB &
#done
# SARSOP tests:
#folder_path="Data/SarsopTest/"
#for env in "ABC" "RockSample5" "FrozenLake4" "Tiger" # QUICK
#do
#    julia --project=. run_sarsoptest.jl --env $env --path $folder_path --discount 0.95 &
#    julia --project=. run_sarsoptest.jl --env $env --path $folder_path --discount 0.99 &
#done

#wait

# echo -e "\n\n============= Large Tests  =============\n\n"
# # 26 tasks, timing capped by TO for SARSOP (default: 3h total)
# folder_path="Data/SarsopTest/"
# for env in "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake10" "Tag" "SparseHallway1" "SparseHallway2" "SparseTigerGrid" # LONG
# do
#     julia --project=. run_sarsoptest.jl --env $env --path $folder_path --discount 0.95 &
#     julia --project=. run_sarsoptest.jl --env $env --path $folder_path --discount 0.99 &
# done

# # UB Tests: at worst about as long as above...
# for env in "K-out-of-N3" "Tag" "SparseHallway1" "SparseHallway2" "SparseTigerGrid"  # LONG
# do
#  julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.95 &
#  julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.99 &
# done
# wait

# echo -e "\n\n============= RUNS COMPLETED =============\n\n"

