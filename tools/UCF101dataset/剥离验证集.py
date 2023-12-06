import shutil, os

txtlist = ['testlist01.txt', 'testlist01.txt', 'testlist01.txt']
dataset_dir = "E:/tmp/UCF101/UCF101/UCF-101/"  # 数据存放路径
copy_path = 'E:/tmp/UCF101/val/'  # 验证集存放路径

for txtfile in txtlist:
	for line in open(txtfile, 'r'):
		o_filename = dataset_dir + line.strip()
		n_filename = copy_path + line.strip()
		if not os.path.exists('/'.join(n_filename.split('/')[:-1])):
			os.makedirs('/'.join(n_filename.split('/')[:-1]))
		try:
			shutil.move(o_filename, n_filename)
			print(o_filename, n_filename)
		except:
			print(o_filename,"不存在")


