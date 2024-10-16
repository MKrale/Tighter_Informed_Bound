#!/bin/bash

# File to run all experiments used in the paper.

processes=()
discount="0.95"



## Small, UB
for env in "ABC" "RockSample5" "FrozenLake4" "Tiger" # QUICK
do
  thisrun="julia --project=. run_upperbound.jl --env $env --discount $discount --precompile true"
  processes+=("$thisrun")
done
## Small, Sarsop
for env in "ABC" "RockSample5" "Tiger" "K-out-of-N2" # QUICK
do
   processes+=("julia --project=. run_sarsoptest.jl --env $env --discount $discount --onlyBs true --precompile true")
done
###Large, UB
for env in "RockSample7" "SparseHallway1" "SparseHallway2" "K-out-of-N2" "K-out-of-N3" "Tag"  "SparseTigerGrid" # LONG
do
 processes+=("julia --project=. run_upperbound.jl --env $env --discount $discount --precompile false")
done
## Large, Sarsop
for env in "SparseHallway1" "SparseHallway2"  "RockSample7" "K-out-of-N2" "K-out-of-N3" "Tag" "SparseTigerGrid" # LONG
do
 processes+=("julia --project=. run_sarsoptest.jl --env $env --discount $discount --onlyBs true --precompile false")
done

### Extra Large:

for env in "pentagon" "grid" "aloha30" "fourth" 
do
 processes+=("julia --project=. run_upperbound.jl --env $env --discount $discount --precompile false")
done
for env in "pentagon" "grid" "aloha30" "fourth" 
do
 processes+=("julia --project=. run_sarsoptest.jl --env $env --discount $discount --onlyBs true --precompile false")
done


folder_path="Data/DiscountTest/"
for env in "Tiger" "FrozenLake4" "K-out-of-N2"
do
   for discount in $(seq 0.95 0.001 0.99);
   do
      processes+=("julia --project=. run_sarsoptest.jl --env $env --path $folder_path --discount $discount")
   done
done
wait



printf "%s\n" "${processes[@]}" | parallel -j3
wait
echo -e "\n\n============= RUNS COMPLETED =============\n\n"