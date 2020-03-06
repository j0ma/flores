all: init train_all evaluate_all

init: install download

install:
	pip install fairseq sacrebleu sentencepiece

download:
	bash download-data.sh
	bash prepare-neen.sh
	bash prepare-sien.sh

train_all:
	# 1. Train NE - EN
	bash ./train.sh "ne" "en"

	# 2. Train EN - NE
	bash ./train.sh "en" "ne"

	# 3. Train SI - EN
	bash ./train.sh "si" "en"

	# 4. Train EN - SI
	bash ./train.sh "en" "si"

evaluate_all:
	# 1. Evaluate NE - EN
	bash evaluate.sh "ne" "en"

	# 2. Evaluate EN - NE
	bash evaluate.sh "en" "ne"

	# 3. Evaluate SI - EN
	bash evaluate.sh "si" "en"

	# 4. Evaluate EN - SI
	bash evaluate.sh "en" "si"

