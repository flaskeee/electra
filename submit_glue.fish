set tasks cola  mnli  mrpc  qnli  qqp  rte  sst  sts  
set model_name $argv[1]

for t in $tasks;
    sbatch run_finetuning.sh $run_name $t $model_name;
end
