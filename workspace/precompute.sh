module load cuda
module load gcc/14.2.0

eval "$(micromamba shell hook --shell bash)"
micromamba activate genomics
cd /fs/nexus-scratch/zli12321/nucleotide-transformer/workspace


export CUDA_VISIBLE_DEVICES=0,1
python precompute_embeddings.py 