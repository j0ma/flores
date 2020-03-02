for file in ../evaluate/results/*_test.txt
do
    echo $file;
    grep -o "BLEU[4]* = [0-9].[0-9][0-9]" $file
done
