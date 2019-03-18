#!/bin/bash
#PBS -l select=1:ncpus=1
#PBS -N qusimu
#PBS -j oe

cd ${PBS_O_WORKDIR}
. /etc/profile.d/modules.sh
# module load cuda ...
. ~/.modules.sh

num_qubits=30
insts_size=5000

echo "num_qubits : $num_qubits"
echo "insts_size : $insts_size"

echo "/////// imp1 //////"
./gen <<< "$num_qubits $insts_size" | time ./imp1/qusimu
echo "/////// imp2 //////"
./gen <<< "$num_qubits $insts_size" | time ./imp2/qusimu
# ./gen <<< "30 500" | nvprof -a global_access,shared_access,branch,instruction_execution,pc_sampling -f -o qusimu.1.nvvp ./qusimu
