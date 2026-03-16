to run the sharded, use:

JOBID=$(sbatch --parsable launch_eval.sh)
echo "Array job: $JOBID"

sbatch --dependency=afterok:$JOBID merge_eval.sh $JOBID