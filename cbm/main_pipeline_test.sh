# 1. give annotations for pope
# load object_dict.pth based on training data
# give labels based on this dict to annotate pope 
# torch.save(dataset, "pope_ce_test_data.pth")

python pope_test_data_trainsform.py

python aokvqa_test_data_trainsform.py

python chair_test_data_trainsform.py

python hal_feature_extract.py
python hal_ms_coco_training_data_trainsform.py

python vqa_feature_extract.py
python vqa_ms_coco_training_data_trainsform.py

python coco_caption_feature_extract.py
python coco_caption_training_data_trainsform.py

# # 2. construct training data
# this use llava to run on pope and get llava features
# torch.save(info_save_list, cbm_path+'pope_info_probe_list.pth')
# torch.save(label_list, cbm_path+'pope_label_list.pth')
# torch.save(output_list, cbm_path+'pope_output_list.pth')  # 新增保存模型输出的代码
# torch.save(question_list, cbm_path+'pope_question_list.pth')  # 新增保存模型输出的代码
# torch.save(vit_feature_list, cbm_path+'pope_vit_feature_list.pth')  # 新增保存模型输出的代码

cd ../LLaVA
./run-pope-cbm.sh

./run-chair-cbm.sh

./run-aokvqa-cbm.sh

./hal_datagen.sh

./vqa_datagen.sh

./coco_caption_datagen.sh

# 3. use these features to train a ce model
# use the feature and the annotation to evaluate information of the llava feature
cd ../cbm

python kmeans.py
python ce_test.py

# python chair_ce_test.py

# python aokvqa_ce_test.py

# python hal_ce_test.py

# python vqa_ce_test.py

# yolo dict: {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}

# result/plot_offline.py

###############################################################################################
python kmeans.py
python coco_caption_ce_test.py

################################VQA v2 on COCO validation#######################################
# python vqa_feature_extract.py
# python vqa_ms_coco_training_data_trainsform.py # training_set=False
# LLaVA/vqa_datagen.py/sh

################################text prompt influence related#######################################
# ./run-pope-cbm.sh

################################CHAIR Open dataset#######################################

# python chair_test_data_trainsform.py

# cd ../LLaVA
# ./run-chair-cbm.sh # with run-chair-cbm-v1.py

python kmeans.py
python chair_ce_test.py

# (kmeans.py)

################################HAL dataset#######################################

# python hal_feature_extract.py
# python hal_ms_coco_training_data_trainsform.py

# LLaVA/hal_datagen.py/sh

python kmeans.py
python hal_ce_test.py
# (kmeans)

############################VQA dataset#########################################

python kmeans.py
python vqa_ce_test.py


############################AOK-VQA dataset#########################################

python kmeans.py
python aokvqa_ce_test.py

###############################Final Fv Calculation #######################################

./classify.sh

###################################################################
###############################domain##############################
###################################################################

vqa_question_type.py
domain_inject_metric.py

python domain_ce_test.py
python vqa_question_type.py

#########################################################################
#########################################################################
#########################################################################
#########################################################################
#########################################################################
### for the Ft calculation

python inject_metric.py
python visual_inject_metric.py
python text_inject_metric.py
python hal_visual_inject_metric.py
python domain_inject_metric.py

###########################

python metric2_plot_offline.py
