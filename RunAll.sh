#!/bin/bash

# File to run all experiments used in the paper. For analysing the created data, see Plotting.ipynb

#### Normal experiments #####

processes=()
discount="0.95"

## Small, UB
for env in "ABC" "RockSample5" "Tiger" "K-out-of-N2" # QUICK, using precompile to minimize variance
do
  thisrun="julia --project=. run_upperbound.jl --env $env --discount $discount --precompile true"
  processes+=("$thisrun")
done
# Small, Sarsop
for env in "ABC" "RockSample5" "Tiger" "K-out-of-N2" # QUICK
do
   processes+=("julia --project=. run_sarsoptest.jl --env $env --discount $discount --onlyBs true --precompile true")
done

###Large, UB
for env in "RockSample7" "SparseHallway1" "SparseHallway2" "K-out-of-N3" "Tag"  "SparseTigerGrid" # LONG, not using precompile to save on computational cost
do
 processes+=("julia --project=. run_upperbound.jl --env $env --discount $discount --precompile false")
done
# Large, Sarsop
for env in "SparseHallway1" "SparseHallway2"  "RockSample7" "K-out-of-N3" "Tag" "SparseTigerGrid" # LONG
do
 processes+=("julia --project=. run_sarsoptest.jl --env $env --discount $discount --onlyBs true --precompile false")
done

### Extra Large:

for env in "pentagon" "grid" "aloha30" "fourth" # LONGER
do
 processes+=("julia --project=. run_upperbound.jl --env $env --discount $discount --precompile false")
done
for env in "pentagon" "grid" "aloha30" "fourth" # LONGER
do
 processes+=("julia --project=. run_sarsoptest.jl --env $env --discount $discount --onlyBs true --precompile false")
done

printf "%s\n" "${processes[@]}" | parallel -j1
wait

##### Discount experiments #####

folder_path="Data/DiscountTest/"
for env in "K-out-of-N2" "RockSample5" # "Tiger"
do
  for heuristic in "standard" "TIB" "ETIB"
  do
    for discount in $(seq 0.95 0.001 0.999);
    do
        start_time=$(date +%s)
        julia --project=. run_sarsoptest.jl --env $env --precompile true --path $folder_path --solvers $heuristic --discount $discount
        end_time=$(date +%s)
        elapsed_time=$((end_time - start_time))
        if [ $elapsed_time -gt 3600 ]; then 
          echo "Run took $elapsed_time longer than 3600s: computations stopped."
          break
        fi
    done
  done
done
wait
echo -e "\n\n============= RUNS COMPLETED =============\n\n"
