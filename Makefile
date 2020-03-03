all: init train evaluate

init:
	pip install fairseq sacrebleu sentencepiece
	bash download-data.sh
	bash prepare-neen.sh
	bash prepare-sien.sh

train:
	CHECKPOINT_DIR=${CHECKPOINT_DIR} ./train.sh

evaluate:
	CHECKPOINT_DIR=${CHECKPOINT_DIR} ./evaluate.sh

