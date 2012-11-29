#!/bin/bash

data=("a1a" "a2a")
methods=(50 54 55)
liblinear_methods=(0 2 5 6)
losses=(1 2 3)
curvs=(0 1 2)
alphas=(0 1)

for d in "${data[@]}"
do
	for m in "${liblinear_methods[@]}"
	do
		../train -B 1 -h 4 -s $m -x ${d}.test ${d}.train ${m}_${d}_0_0_0_model.txt > ${m}_${d}_0_0_0_out.txt
	done
	for m in "${methods[@]}"
	do
		for l in "${losses[@]}"
		do
			for c in "${curvs[@]}"
			do
				for a in "${alphas[@]}"
				do
					../train -B 1 -h 4 -s $m -u $c -l $l -r $a -x ${d}.test ${d}.train ${m}_${d}_${l}_${c}_${a}_model.txt > ${m}_${d}_${l}_${c}_${a}_out.txt
				done
			done
		done
	done
done
