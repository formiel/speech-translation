#!/bin/bash
# Copyright 2019 Hang Le (hangtp.le@gmail.com)

# Argument 1 is number of jobs, argument 2 is the config file
if [ $# -gt 1 ]; then
    Njobs=$1
    CONFIG=$2
    echo "Number of jobs (excluding the first one): ${Njobs}"
    echo "Configuration file: ${CONFIG}"
else
    echo "Require 2 arguments!!!!!!! Exit."
    exit
fi

# first job - no dependencies
j0=$(sbatch run.slurm $CONFIG)
# Format of j0: "Submitted job <jobID>". Here we get the true jobID
j0_id=$(echo $j0 | sed 's/[^0-9]*//g')
echo "ID of the first job: $j0_id"

# add first job to the list of jobs
jIDs+=($j0_id)

# for loop: submit Njobs: where job i + 1 is dependent on job i.
# and job i + 1 resume from the checkpoint of job i
for i in $(seq 1 $Njobs); do
    # Submit job i+1 (i.e. new_job) with dependency ('afterok:') on job i
    new_job=$(sbatch --dependency=afterok:${jIDs[ $i - 1 ]} run.slurm $CONFIG)
    new_job_id=$(echo $new_job | sed 's/[^0-9]*//g')
    echo "Submitted Job ID $new_job_id that will be executed once Job ID ${jIDs[ $i - 1 ]} has completed with success"
    jIDs+=($new_job_id)
done
echo "List of jobs that have been submitted: ${jIDs[@]}"