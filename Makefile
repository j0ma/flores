all: init train evaluate

init: install download

install:
	pip install fairseq sacrebleu sentencepiece

download:
	bash download-data.sh
	bash prepare-neen.sh
	bash prepare-sien.sh

train:

    TIME_SUFFIX=$(date -Iminutes | sed s/':'/'-'/g)
	export LOG_FOLDER="./LOG/"$TIME_SUFFIX
	mkdir -p $LOG_FOLDER

	# 1. Train NE - EN
	bash ./train.sh "ne" "en" $LOG_FOLDER

	# 2. Train EN - NE
	bash ./train.sh "en" "ne" $LOG_FOLDER

	# 3. Train SI - EN
	bash ./train.sh "si" "en" $LOG_FOLDER

	# 4. Train EN - SI
	bash ./train.sh "en" "si" $LOG_FOLDER


evaluate:

    export TIME_SUFFIX=$(date -Iminutes | sed s/":"/"-"/g)
	export RESULTS_FOLDER="./evaluate/"$TIME_SUFFIX
	mkdir -p $RESULTS_FOLDER

	# 1. Evaluate NE - EN
	bash evaluate.sh "ne" "en" $RESULTS_FOLDER

	# 2. Evaluate EN - NE
	bash evaluate.sh "en" "ne" $RESULTS_FOLDER

	# 3. Evaluate SI - EN
	bash evaluate.sh "si" "en" $RESULTS_FOLDER 

	# 4. Evaluate EN - SI
	bash evaluate.sh "en" "si" $RESULTS_FOLDER 

