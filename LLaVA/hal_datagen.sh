# cd transformers
# git checkout v4.37.2

temperature=0.7

# # tmux 1
CUDA_VISIBLE_DEVICES=0 python hal_datagen.py \
        --model-path "liuhaotian/llava-v1.5-7b" --question "correct" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt
# [32, 34, 130, 266, 474, 517, 585, 655, 703, 846, 993, 1073, 1124, 1346, 1388, 1511, 1547, 1589, 1622, 1741, 1874, 1885, 1914, 1989, 2019, 2021, 2024, 2151, 2175, 2260, 2265, 2403, 2407, 2528, 2573, 2622, 2798, 2803, 2950, 3009, 3190, 3331, 3430, 3464, 3538, 3565, 3570, 3587, 3589, 3591, 3602, 3605, 3727, 3729, 3732, 3801, 3836, 3840, 3843, 3946, 3965, 4075, 4117, 4243, 4350, 4353, 4356, 4424, 4454, 4455, 4484, 4603, 4612, 4710, 4731, 4852, 4854]

# --question "correct"
# --question "correctorigin"
# --question "correctvisual_prompt"
# --question "correctvisual_prompt-1"
# --question "correctvisual_prompt-2"

CUDA_VISIBLE_DEVICES=0 python hal_datagen.py \
        --model-path "liuhaotian/llava-v1.5-7b" --question "wrong" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" | tee vaq_train_gen_log-1.txt
# [128, 129, 130, 131, 136, 137, 138, 139, 520, 521, 522, 523, 1064, 1065, 1066, 1067, 1896, 1897, 1898, 1899, 2068, 2069, 2070, 2071, 2340, 2341, 2342, 2343, 2620, 2621, 2622, 2623, 2812, 2813, 2814, 2815, 3384, 3385, 3386, 3387, 3972, 3973, 3974, 3975, 4292, 4293, 4294, 4295, 4496, 4497, 4498, 4499, 5384, 5385, 5386, 5387, 5552, 5553, 5554, 5555, 6044, 6045, 6046, 6047, 6188, 6189, 6190, 6191, 6356, 6357, 6358, 6359, 6488, 6489, 6490, 6491, 6964, 6965, 6966, 6967, 7496, 7497, 7498, 7499, 7540, 7541, 7542, 7543, 7656, 7657, 7658, 7659, 7956, 7957, 7958, 7959, 8076, 8077, 8078, 8079, 8084, 8085, 8086, 8087, 8096, 8097, 8098, 8099, 8604, 8605, 8606, 8607, 8700, 8701, 8702, 8703, 9040, 9041, 9042, 9043, 9060, 9061, 9062, 9063, 9612, 9613, 9614, 9615, 9628, 9629, 9630, 9631, 10112, 10113, 10114, 10115, 10292, 10293, 10294, 10295, 10488, 10489, 10490, 10491, 11192, 11193, 11194, 11195, 11212, 11213, 11214, 11215, 11800, 11801, 11802, 11803, 12036, 12037, 12038, 12039, 12760, 12761, 12762, 12763, 13324, 13325, 13326, 13327, 13720, 13721, 13722, 13723, 13856, 13857, 13858, 13859, 14152, 14153, 14154, 14155, 14260, 14261, 14262, 14263, 14280, 14281, 14282, 14283, 14348, 14349, 14350, 14351, 14356, 14357, 14358, 14359, 14364, 14365, 14366, 14367, 14408, 14409, 14410, 14411, 14420, 14421, 14422, 14423, 14908, 14909, 14910, 14911, 14916, 14917, 14918, 14919, 14928, 14929, 14930, 14931, 15204, 15205, 15206, 15207, 15344, 15345, 15346, 15347, 15360, 15361, 15362, 15363, 15372, 15373, 15374, 15375, 15784, 15785, 15786, 15787, 15860, 15861, 15862, 15863, 16300, 16301, 16302, 16303, 16468, 16469, 16470, 16471, 16972, 16973, 16974, 16975, 17400, 17401, 17402, 17403, 17412, 17413, 17414, 17415, 17424, 17425, 17426, 17427, 17696, 17697, 17698, 17699, 17816, 17817, 17818, 17819, 17820, 17821, 17822, 17823, 17936, 17937, 17938, 17939, 18412, 18413, 18414, 18415, 18448, 18449, 18450, 18451, 18840, 18841, 18842, 18843, 18924, 18925, 18926, 18927, 19408, 19409, 19410, 19411, 19416, 19417, 19418, 19419]

# --question "wrong"
# --question "wrongorigin"
# --question "wrongvisual_prompt-1"
# --question "wrongvisual_prompt-2"
# --question "wrongvisual"

###########################################################################

CUDA_VISIBLE_DEVICES=0 python hal_datagen.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "correctorigin" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature"  #| tee vaq_train_gen_log-1.txt

# --question "correctorigin"
# --question "correctvisual_prompt"
# --question "correctvisual_prompt-1"
# --question "correctvisual_prompt-2"

CUDA_VISIBLE_DEVICES=0 python hal_datagen.py \
        --model-path "Qwen/Qwen2.5-VL-3B-Instruct" --question "wrongorigin" --answers-file "$answer_path" --image-folder "$image_folder" --temperature "$temperature" # | tee vaq_train_gen_log-1.txt

# --question "wrongorigin"
# --question "wrongvisual_prompt"
# --question "wrongvisual_prompt-1"
# --question "wrongvisual_prompt-2"
