#!/bin/bash

# ... description ...
discount="0.95"
processes=()
# Small, UB
for env in "ABC" "RockSample5" "FrozenLake4" "Tiger" # QUICK
do
   thisrun="julia --project=. run_upperbound.jl --env $env --discount $discount"
   processes+=("$thisrun")
done
# Small, Sarsop
# for env in "ABC" "RockSample5" "FrozenLake4" "Tiger" # QUICK
# do
#    processes+=("julia --project=. run_sarsoptest.jl --env $env --discount $discount")
# done
# # Large, UB
# for env in "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake10" "Tag" "SparseHallway1" "SparseHallway2" "SparseTigerGrid" # LONG
# do
#    processes+=("julia --project=. run_upperbound.jl --env $env --discount $discount")
# done
# for env in "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake10" "Tag" "SparseHallway1" "SparseHallway2" "SparseTigerGrid" # LONG
# do
#    processes+=("julia --project=. run_sarsoptest.jl --env $env --discount $discount")
# done

printf "%s\n" "${processes[@]}" | parallel -j3 # WHY DOES THIS WORK??? I HATE BASH!!!
wait
echo -e "\n\n============= RUNS COMPLETED =============\n\n"




# echo -e "\n\n============= Quick Tests (UB & SARSOP)  =============\n\n"
# # 16 tasks, generally takes < 10 min to run

# ### Upper bounds:
# folder_path="Data/UpperBounds/"
# for env in "ABC" "RockSample5" "FrozenLake4" "Tiger" # QUICK
# ## NOTE: Frozen Lake 10 at d=0.99 does not compile correctly...
# do
#    julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.95 &
#    # julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.99 &
# done
# ### SARSOP tests:
# folder_path="Data/SarsopTest/"
# for env in "ABC" "RockSample5" "FrozenLake4" "Tiger" # QUICK
# do
#    julia --project=. run_sarsoptest.jl --env $env --path $folder_path --discount 0.95 &
#    # julia --project=. run_sarsoptest.jl --env $env --path $folder_path --discount 0.99 &
#    julia --project=. run_sarsoptest.jl --env $env --path $folder_path --solvers WBIB --discount 0.95 &
#    # julia --project=. run_sarsoptest.jl --env $env --path $folder_path --solvers WBIB --discount 0.99 &
# done
# wait

# # echo -e "\n\n============= Large Tests  =============\n\n"
# # ### 32 tasks, timing capped by TO ( = max(5xTO ub, 3xTO SARSOP) = 3h default )

# # # Upper bounds:
# folder_path="Data/UpperBounds/"
# for env in "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake10" "Tag" "SparseHallway1" "SparseHallway2" "SparseTigerGrid" # LONG
# do
#    julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.95 &
#    # julia --project=. run_upperbound.jl --env $env --path $folder_path --discount 0.99 &
# done
# # ### SARSOP tests:
# folder_path="Data/SarsopTest/"
# for env in "RockSample10" "K-out-of-N2" "K-out-of-N3" "FrozenLake10" "Tag" "SparseHallway1" "SparseHallway2" "SparseTigerGrid" # LONG
# do
#    julia --project=. run_sarsoptest.jl --env $env --path $folder_path --discount 0.95 &
#    # julia --project=. run_sarsoptest.jl --env $env --path $folder_path --discount 0.99&
# done
# wait

# # ### Discount tests:
# # folder_path="Data/SarsopTest/"
# # for env in "Tiger"
# # do
# #    for factor in {-6..20}
# #       discount=$((1-(0.1*0.9**$factor)))
# #       do
# #          julia --project=. run_sarsoptest.jl --env $env --path $folder_path --discount $discount &
# #       done
# # done
# # wait

# echo -e "\n\n============= RUNS COMPLETED =============\n\n"

