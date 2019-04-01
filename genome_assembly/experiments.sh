# file=smaller/small.txt
# for value in "${core[@]}"
# do
#         echo 'n = ' $value
#         srun -N 1 -n $value ./kmer_hash $dna_path$file
# done

# # comment the test if not needed

# echo '------19-kmer little file test result------'

# file=smaller/little.txt
# for value in "${core[@]}"
# do
#         echo 'n = ' $value
#         srun -N 1 -n $value ./kmer_hash $dna_path$file
# done


# echo '------19-kmer tiny file test result------'

# file=smaller/tiny.txt
# for value in "${core[@]}"
# do
#         echo 'n = ' $value
#         srun -N 1 -n $value ./kmer_hash $dna_path$file
# done

# echo '------19-kmer test.txt file test result-------'

# file=test.txt
# for value in "${core[@]}"
# do
# 	echo 'n = ' $value
# 	srun -N 1 -n $value ./kmer_hash $dna_path$file
# done

dna_path=/global/project/projectdirs/mp309/cs267-spr2018/hw3-datasets/
core=(1 2 4 8 16 32)
node={1 2 4 8}


echo '------Strong Scaling: 51-kmer large.txt file test result-------'

file=large.txt
for value in "${core[@]}"
do
	echo 'n = ' $value
	export UPCXX_SEGMENT_MB=256
	export GASNET_MAX_SEGSIZE=8GB
	srun -N 1 -n $value ./kmer_hash $dna_path$file
done


echo '------Strong Scaling: 51-kmer human-chr14-synthetic.txt file test result-------'

file=human-chr14-synthetic.txt
for value in "${core[@]}"
do
	echo 'n = ' $value
	export UPCXX_SEGMENT_MB=256
	export GASNET_MAX_SEGSIZE=8GB
	srun -N 1 -n $value ./kmer_hash $dna_path$file
done


echo '------Weak Scaling: 51-kmer large.txt file test result-------'

file=large.txt
for value in "${node[@]}"
do
	echo 'N = ' $value
	srun -N $value -n 32 ./kmer_hash $dna_path$file
done


echo '------Weak Scaling: 51-kmer human-chr14-synthetic.txt file test result-------'

file=human-chr14-synthetic.txt
for value in "${node[@]}"
do
	echo 'N = ' $value
	srun -N $value -n 32 ./kmer_hash $dna_path$file
done





















