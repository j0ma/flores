# recover_results.sh

bash recover_parameters.sh $1

echo "===== RESULTS ====="
for log_file in $1/*.log;
do
   LANG_PAIR=$(echo $log_file | \
               grep -o 'baseline_.*.log' | \
               sed "s/baseline_//g" | \
               sed "s/\.log//g" | \
               sed "s/_/-/g")
   BLEU=$(grep -Eo 'BLEU4* = [0-9]\.[0-9]+' $log_file | \
          grep -Eo "[0-9]\.[0-9]+")
   echo $LANG_PAIR "|" $BLEU
done
