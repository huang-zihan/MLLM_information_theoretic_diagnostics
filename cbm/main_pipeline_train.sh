# 1. run feature_extract
# extract annotations for COCO2014
#torch.save(object_dict, 'object_dict_full.pth')
# torch.save(raw_result, 'raw_result_full.pth')

# [apple :0.5, 0.6, dog:0.3] # average
# 1.1 0.3
python feature_extract.py


# 2. construct training data
# torch.save(dataset, "ce_data.pth") with image feature and softmaxed feature ground truth
# for running llava and getting llava feature
python ms_coco_training_data_trainsform.py

# 用 500 sample 计算F1

# 3. cd to the LLaVa and extract llava features based on "ce_data.pth"
# torch.save(info_save_list, ce_data_dir+'ce_training.pth')
# torch.save(label_list, ce_data_dir+'ce_training_label.pth')
# torch.save(output_list, ce_data_dir+'ce_training_response.pth')  # 新增保存模型输出的代码

cd ../LLaVA
./ce_datagen.sh

4. use these features to train a ce model
# torch.save(model.state_dict(), f'ce_model_{index}.pth')
cd ../cbm
./ce_train.sh

##############################

# vqa_ms_coco_training_data_trainsform-v1.py # training_set=True


