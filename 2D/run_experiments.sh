#!/bin/bash

ITERATIONS=10

for iter in `seq 1 $ITERATIONS`;
do

        python3 collect_examples.py 
        python3 summarise_examples.py

        ## EXP 1 - Flat
        python3 collect_results.py --exp 1 --models "test" --examples 0
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 1 --models "test" --examples 1
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 1 --models "test" --examples 2
        python3 collect_examples.py 
        python3 summarise_examples.py

        git add ./results/
        git commit -m "Results Update"
        git push

        ## EXP 8 - Flat Baseline
        python3 collect_results.py --exp 8 --models "test" --examples 0
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 8 --models "test" --examples 1
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 8 --models "test" --examples 2
        python3 collect_examples.py 
        python3 summarise_examples.py

        git add ./results/
        git commit -m "Results Update"
        git push

        ## EXP 2 - Curved
        python3 collect_results.py --exp 2 --models "test" --examples 0
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 2 --models "test" --examples 1
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 2 --models "test" --examples 2
        python3 collect_examples.py 
        python3 summarise_examples.py

        git add ./results/
        git commit -m "Results Update"
        git push

        ## EXP 9 - Curved Baseline
        python3 collect_results.py --exp 9 --models "test" --examples 0
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 9 --models "test" --examples 1
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 9 --models "test" --examples 2
        python3 collect_examples.py 
        python3 summarise_examples.py

        git add ./results/
        git commit -m "Results Update"
        git push

        ### EXP 11 - Difficulty Baseline 
        python3 collect_results.py --exp 11 --models "test" --examples 0
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 11 --models "test" --examples 1
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 11 --models "test" --examples 2
        python3 collect_examples.py 
        python3 summarise_examples.py

        git add ./results/
        git commit -m "Results Update"
        git push

        # EXP 10 - Difficulty No baseline 
        python3 collect_results.py --exp 10 --models "test" --examples 0
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 10 --models "test" --examples 1
        python3 collect_examples.py 
        python3 summarise_examples.py

        python3 collect_results.py --exp 10 --models "test" --examples 2
        python3 collect_examples.py 
        python3 summarise_examples.py

        git add ./results/
        git commit -m "Results Update"
        git push

done

