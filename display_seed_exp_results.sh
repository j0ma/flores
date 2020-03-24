for f in $(ls -t ./log/ | head -n 10)
do
    top_row=$(head -n 1 ./log/$f/baseline_en_ne.log)
    s=$(echo $top_row | grep -o "SEED=1[0-9]")
    echo "$f -> $s"
done
