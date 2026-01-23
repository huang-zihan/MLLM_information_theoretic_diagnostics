import matplotlib.pyplot as plt

# classify_type="cifar100"
# classify_type="cifar10"
# classify_type="mnist"
classify_type='pope'
# classify_type='pope-global'
# classify_type='cross'
# classify_type='counter'
# classify_type='mismatch'
# mask=True
mask=False
vit=True

# 读取文件
results = {}

if classify_type=="mnist":
    written_file="training_results.txt"
    pic_dir='model_accuracy.pdf'
elif classify_type=="cifar100":
    if mask:
        written_file="training_results_cifar100_mask.txt"
        pic_dir='model_accuracy_cifar100_mask.pdf'
    else:
        written_file="training_results_cifar100.txt"
        pic_dir='model_accuracy_cifar100.pdf'
elif classify_type=='pope':
    # written_file="pope_result.txt"
    # pic_dir='pope_f1_result.pdf'
    
    written_file="globalpope_result.txt"
    pic_dir='globalpope_f1_result.pdf'
    
    # written_file="pope_result_dog.txt"
    # pic_dir='pope_acc_result_dog.pdf'
    
        
    # written_file="pope_result_chair.txt"
    # pic_dir='pope_acc_result_chair.pdf'
    
    # written_file="pope_result_keyboard.txt"
    # pic_dir='pope_acc_result_keyboard.pdf'
    
    # written_file="pope_result_car.txt"
    # pic_dir='pope_acc_result_car.pdf'
elif classify_type=='counter':
    # written_file="pope_result_chair_counter.txt"
    # pic_dir='pope_acc_result_chair_counter.pdf'
    written_file="pope_result_backpack_counter.txt"
    pic_dir='pope_acc_result_backpack_counter.pdf'
elif classify_type=='mismatch':
    written_file="pope_resultdog-keyboard-mismatch.txt"
    pic_dir='pope_acc_result_dog-keyboard_mismatch.pdf'
elif classify_type=='cross':
    # item='car'
    item='dining table'
    # ques='dining table'
    ques='car'
    written_file=f"pope_result{item}-{ques}.txt"
    pic_dir=f'pope_acc_result_{item}-{ques}.pdf'
elif classify_type=='pope-global':
    written_file="globalpope_result.txt"
    pic_dir='globalpope_f1_result.pdf'
else:
    if mask:
        written_file="training_results_cifar10_mask.txt"
        pic_dir='model_accuracy_cifar10_mask.pdf'
    else:
        written_file="training_results_cifar10.txt"
        pic_dir='model_accuracy_cifar10.pdf'

if vit:
    with open('vit_vit_pope_result.txt', "r") as f:
        next(f)  # 跳过标题行
        for line in f:
            index, epoch, loss, accuracy = line.strip().split('\t')
            index = int(index)
            epoch = int(epoch)
            accuracy = float(accuracy)
            index=-1
            if index not in results:
                results[index] = {'epochs': [], 'accuracies': []}
            
            results[index]['epochs'].append(epoch)
            results[index]['accuracies'].append(accuracy)

with open(written_file, "r") as f:
    next(f)  # 跳过标题行
    for line in f:
        index, epoch, loss, accuracy = line.strip().split('\t')
        index = int(index)
        epoch = int(epoch)
        accuracy = float(accuracy)

        if index not in results:
            results[index] = {'epochs': [], 'accuracies': []}
        
        results[index]['epochs'].append(epoch)
        results[index]['accuracies'].append(accuracy)


# plot_index=[0,12,24,31]
plot_index=[-1, 0, 12, 24, 27, 30, 31]
# 绘制准确性图像
plt.figure(figsize=(6, 4))
for index, metrics in results.items():
    if index not in plot_index:
        continue
    if index==-1:
        plt.plot(metrics['epochs'], metrics['accuracies'], label='ViT', linestyle='--')  # 使用虚线和蓝色
    else:
        plt.plot(metrics['epochs'], metrics['accuracies'], label=f'Index {index}')

plt.title('Model Accuracy per Epoch')
plt.xlabel('Epoch')
if 'pope' in classify_type or 'cross' in classify_type:
    plt.ylabel('F1')
else:
    plt.ylabel('Accuracy (%)')
# plt.xticks(range(1, max(metrics['epochs']) + 1))
# if classify_type=='cifar10':
#     if mask:
#         plt.ylim(60, 90)
#     else:
#         plt.ylim(85, 100)
# plt.ylim(85, 100)
# elif classify_type=='cifar100':
#     if mask:
#         plt.ylim(40, 65)
#     else:
#         plt.ylim(60, 90)
    
plt.legend()
plt.grid()
plt.tight_layout()
# plt.show()

# 保存图像到文件
# plt.savefig(pic_dir)
plt.savefig(pic_dir, format='pdf')
plt.close()  # 关闭图像以释放内存


