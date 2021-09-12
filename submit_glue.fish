set tasks cola  mnli  mrpc  qnli  qqp  rte  sst  sts  
set run_name $argv[1]
set model_name $argv[2]
mkdir runs/$run_name
for t in $tasks;
    sbatch run_finetuning.sh $run_name $t $model_name;
end
