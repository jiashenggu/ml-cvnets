 export CFG_FILE="config/classification/imagenet/mobilevit_v2.yaml"
 export MODEL_WEIGHTS="/home/gujiashe/ml-cvnets/models/mobilevitv2-2.0.pt"
 export RESULTS="results"
 CUDA_VISIBLE_DEVICES=0 cvnets-eval --common.config-file $CFG_FILE --common.results-loc $RESULTS --model.classification.pretrained $MODEL_WEIGHTS