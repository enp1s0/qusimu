#!/bin/bash
#PBS -l select=1:ncpus=1
#PBS -N qusimu
#PBS -j oe

cd ${PBS_O_WORKDIR}
. /etc/profile.d/modules.sh
# module load cuda ...
. ~/.modules.sh

num_qubits=25
insts_size=5000

echo "num_qubits : $num_qubits"
echo "insts_size : $insts_size"

echo "qusimu"
./gen <<< "$num_qubits $insts_size" | time ./qusimu
echo "// mori-cpu"
./gen <<< "$num_qubits $insts_size" | time ./mori-cpu
# ./gen <<< "30 500" | nvprof -a global_access,shared_access,branch,instruction_execution,pc_sampling -f -o qusimu.1.nvvp ./qusimu
