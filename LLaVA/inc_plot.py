import matplotlib.pyplot as plt

classify_type="cifar100"
# classify_type="cifar10"
# classify_type="mnist"
# mask=True
mask=False

# 读取文件
results = {}

written_file="incorrect_test.txt"
pic_dir='model_accuracy_cifar100_inc.png'
    

# if classify_type=="mnist":
#     written_file="training_results.txt"
#     pic_dir='model_accuracy.png'
# elif classify_type=="cifar100":
#     if mask:
#         written_file="training_results_cifar100_mask.txt"
#         pic_dir='model_accuracy_cifar100_mask.png'
#     else:
#         written_file="training_results_cifar100.txt"
#         pic_dir='model_accuracy_cifar100.png'
# else:
#     if mask:
#         written_file="training_results_cifar10_mask.txt"
#         pic_dir='model_accuracy_cifar10_mask.png'
#     else:
#         written_file="training_results_cifar10.txt"
#         pic_dir='model_accuracy_cifar10.png'
    
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

# 绘制准确性图像
plt.figure(figsize=(10, 6))
for index, metrics in results.items():
    plt.plot(metrics['epochs'], metrics['accuracies'], label=f'Index {index}')

plt.title('Model Accuracy per Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.xticks(range(1, max(metrics['epochs']) + 1))
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
plt.savefig(pic_dir)
plt.close()  # 关闭图像以释放内存


