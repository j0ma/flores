init: install download

install:
	pip install fairseq sacrebleu sentencepiece

download:
	bash download-data.sh
	bash prepare-neen.sh
	bash prepare-sien.sh

train_all_fp16_batch20k:

	# 0. create log & checkpoint folder
	bash ./create_log_folder.sh
	bash ./create_checkpoint_folder.sh

	# 1. Train NE - EN
	bash ./train_fp16_batch20k.sh "ne" "en"

	# 2. Train EN - NE
	bash ./train_fp16_batch20k.sh "en" "ne"

	# 3. Train SI - EN
	bash ./train_fp16_batch20k.sh "si" "en"

	# 4. Train EN - SI
	bash ./train_fp16_batch20k.sh "en" "si"

train_all_fp16_largebatch_seed12345:

	# 0. create log & checkpoint folder
	bash ./create_log_folder.sh
	bash ./create_checkpoint_folder.sh

	# 1. Train NE - EN
	bash ./train_fp16_largebatch_seed12345.sh "ne" "en"

	# 2. Train EN - NE
	bash ./train_fp16_largebatch_seed12345.sh "en" "ne"

	# 3. Train SI - EN
	bash ./train_fp16_largebatch_seed12345.sh "si" "en"

	# 4. Train EN - SI
	bash ./train_fp16_largebatch_seed12345.sh "en" "si"

train_all_fp16_largebatch_lr7e-4:

	# 0. create log & checkpoint folder
	bash ./create_log_folder.sh
	bash ./create_checkpoint_folder.sh

	# 1. Train NE - EN
	bash ./train_fp16_largebatch_lr7e-4.sh "ne" "en"

	# 2. Train EN - NE
	bash ./train_fp16_largebatch_lr7e-4.sh "en" "ne"

	# 3. Train SI - EN
	bash ./train_fp16_largebatch_lr7e-4.sh "si" "en"

	# 4. Train EN - SI
	bash ./train_fp16_largebatch_lr7e-4.sh "en" "si"

train_all_fp16_largebatch_lr5e-4:

	# 0. create log & checkpoint folder
	bash ./create_log_folder.sh
	bash ./create_checkpoint_folder.sh

	# 1. Train NE - EN
	bash ./train_fp16_largebatch_lr5e-4.sh "ne" "en"

	# 2. Train EN - NE
	bash ./train_fp16_largebatch_lr5e-4.sh "en" "ne"

	# 3. Train SI - EN
	bash ./train_fp16_largebatch_lr5e-4.sh "si" "en"

	# 4. Train EN - SI
	bash ./train_fp16_largebatch_lr5e-4.sh "en" "si"

train_all_fp16_largebatch:

	# 0. create log & checkpoint folder
	bash ./create_log_folder.sh
	bash ./create_checkpoint_folder.sh

	# 1. Train NE - EN
	bash ./train_fp16_largebatch.sh "ne" "en"

	# 2. Train EN - NE
	bash ./train_fp16_largebatch.sh "en" "ne"

	# 3. Train SI - EN
	bash ./train_fp16_largebatch.sh "si" "en"

	# 4. Train EN - SI
	bash ./train_fp16_largebatch.sh "en" "si"

train_all_fp16:

	# 0. create log & checkpoint folder
	bash ./create_log_folder.sh
	bash ./create_checkpoint_folder.sh

	# 1. Train NE - EN
	bash ./train_fp16.sh "ne" "en"

	# 2. Train EN - NE
	bash ./train_fp16.sh "en" "ne"

	# 3. Train SI - EN
	bash ./train_fp16.sh "si" "en"

	# 4. Train EN - SI
	bash ./train_fp16.sh "en" "si"

train_all:

	# 0. create log & checkpoint folder
	bash ./create_log_folder.sh
	bash ./create_checkpoint_folder.sh

	# 1. Train NE - EN
	bash ./train.sh "ne" "en"

	# 2. Train EN - NE
	bash ./train.sh "en" "ne"

	# 3. Train SI - EN
	bash ./train.sh "si" "en"

	# 4. Train EN - SI
	bash ./train.sh "en" "si"


evaluate_all:

	# 0. create results folder
	bash ./create_results_folder.sh

	# 1. Evaluate NE - EN
	bash evaluate.sh "ne" "en"

	# 2. Evaluate EN - NE
	bash evaluate.sh "en" "ne"

	# 3. Evaluate SI - EN
	bash evaluate.sh "si" "en"

	# 4. Evaluate EN - SI
	bash evaluate.sh "en" "si"

