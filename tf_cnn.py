# -- coding: utf-8 --
import loader
import tensorflow as tf
import numpy as np

def variable_with_weight_loss(shape, wl):
	# variable_with_weight_loss函数创建卷积核的参数并初始化，shape=[核长，核宽，颜色通道，核数量]
	# wl 即L2规范中lamda值
	# 初始化正太分布的w
	var = tf.Variable(tf.truncated_normal(shape, mean=0, stddev=1))
	if wl is not None:
		# 计算sum(w**2)*(lamda/2)
		weight_loss = tf.multiply(tf.nn.l2_loss(var),wl,name='weight_loss')
		tf.add_to_collection('losses', weight_loss)
	return var

def loss_fun(logits, labels):
	# loss_mean = tf.losses.mean_squared_error(logits, labels)
	# log_loss：\sum(y*ln(y_pred) + (1-y)*ln(1-y_pred)) / num
	loss_mean = tf.losses.log_loss(labels, logits)
	tf.add_to_collection('losses', loss_mean)
	return tf.add_n(tf.get_collection('losses'), name='total_loss')

def get_params(is_test=True):
	if is_test:
		max_steps = 30
		num = 2
		batch_size = 100 * num # num倍数
	else:
		max_steps = 200
		num = 2
		batch_size = 120 * num # num倍数

	labelsDict, labelsIdList, labelsNameList = loader.get_labels_attr() # 获取230个标签ID对应的30个特征
	trainDict, trainPathList, trainNameSet = loader.get_train_id() # 获取训练图片对应的标签ID
	# res_images, label_string, label_number = loader.train_for_test_inputs(trainPathList[39788: ], trainDict, labelsDict) # for test
	# np.save("res_images_test.npy",res_images) # 将数据保存成npy以便使用
	res_images = np.load("res_images_test.npy")
	label_string = np.load("label_string_test.npy")
	label_number = np.load("label_number_test.npy")
	if is_test:
		trainPathList = loader.random_train_path(trainPathList[0: 39788], trainDict)
	else:
		trainPathList = loader.random_train_path(trainPathList, trainDict)


	# 定义输入输出变量
	image_holder = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])
	label_holder = tf.placeholder(tf.float32, [batch_size, 30])

	# 卷积层 - ReLU - 池化层
	weight1 = variable_with_weight_loss(shape=[5, 5, 3, 64], wl=0.0)
	kernel1 = tf.nn.conv2d(image_holder, weight1, [1,1,1,1], padding='SAME')
	bias1 = tf.Variable(tf.constant(0.0, shape=[64]))
	conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
	pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75) # LRN层处理

	# 卷积层 - ReLU - 池化层
	weight2 = variable_with_weight_loss(shape=[5, 5, 64, 64], wl=0.0)
	kernel2 = tf.nn.conv2d(norm1, weight2, [1,1,1,1], padding='SAME')
	bias2 = tf.Variable(tf.constant(0.1, shape=[64]))
	conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75) # LRN层处理
	pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

	# 卷积层 - ReLU - 池化层
	weight3 = variable_with_weight_loss(shape=[5, 5, 64, 64], wl=0.0)
	kernel3 = tf.nn.conv2d(pool2, weight3, [1,1,1,1], padding='SAME')
	bias3 = tf.Variable(tf.constant(0.1, shape=[64]))
	conv3 = tf.nn.relu(tf.nn.bias_add(kernel3, bias3))
	pool3 = tf.nn.max_pool(conv3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')

	# 在卷积层之后使用一个全连接层
	reshape = tf.reshape(pool3, [batch_size, -1])
	dim = reshape.get_shape()[1].value

	# 设置全连接层神经元数量为384
	weight4 = variable_with_weight_loss(shape=[dim, 384], wl=0.002)
	bias4 = tf.Variable(tf.constant(0.1, shape=[384]))
	local4 = tf.nn.relu(tf.matmul(reshape, weight4) + bias4)

	# 设置全连接层神经元数量为192
	weight5 = variable_with_weight_loss(shape=[384, 192], wl=0.002)
	bias5 = tf.Variable(tf.constant(0.1, shape=[192]))
	local5 = tf.nn.relu(tf.matmul(local4, weight5) + bias5)

	# 设置输出层
	weight6 = variable_with_weight_loss(shape=[192, 30], wl=0)
	bias6 = tf.Variable(tf.constant(0.0, shape=[30]))
	logits = tf.add(tf.matmul(local5, weight6), bias6) # 输出值
	logits = tf.nn.sigmoid(logits)

	loss = loss_fun(logits, label_holder) # 计算代价函数
	train_op = tf.train.AdamOptimizer(5e-2).minimize(loss) # 应用Adam优化算法


	sess = tf.Session()
	ini_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	sess.run(ini_op)

	for step in range(max_steps):
		image_batch, label_batch = loader.train_inputs(trainPathList, trainDict, labelsDict, batch_size/num)
		_, loss_value = sess.run([train_op, loss], feed_dict={image_holder: image_batch, label_holder:label_batch})
		if (step + 1) % 5 == 0:
			print(step + 1, loss_value)
			params = sess.run([weight1, bias1, weight2, bias2, weight3, bias3, weight4, bias4, weight5, bias5, weight6, bias6])
			get_result(params, True, res_images, label_string, label_number)

	params = sess.run([weight1, bias1, weight2, bias2, weight3, bias3, weight4, bias4, weight5, bias5, weight6, bias6])
	res_ans, res_names = get_result(params, False)

	logits = sess.run(logits, feed_dict={image_holder: image_batch})
	sess.close()
	return res_ans, res_names, logits



def get_result(params, is_test, res_images=None, label_string=None, label_number=None):
	weight1, bias1, weight2, bias2, weight3, bias3, weight4, bias4, weight5, bias5, weight6, bias6 = params

	batch_size = 1
	image_holder = tf.placeholder(tf.float32, [batch_size, 64, 64, 3])

	# 卷积层 - ReLU - 池化层
	kernel1 = tf.nn.conv2d(image_holder, weight1, [1,1,1,1], padding='SAME')
	conv1 = tf.nn.relu(tf.nn.bias_add(kernel1, bias1))
	pool1 = tf.nn.max_pool(conv1, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
	norm1 = tf.nn.lrn(pool1, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
	# 卷积层 - ReLU - 池化层
	kernel2 = tf.nn.conv2d(norm1, weight2, [1,1,1,1], padding='SAME')
	conv2 = tf.nn.relu(tf.nn.bias_add(kernel2, bias2))
	norm2 = tf.nn.lrn(conv2, 4, bias=1.0, alpha=0.001/9.0, beta=0.75)
	pool2 = tf.nn.max_pool(norm2, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
	# 卷积层 - ReLU - 池化层
	kernel3 = tf.nn.conv2d(pool2, weight3, [1,1,1,1], padding='SAME')
	conv3 = tf.nn.relu(tf.nn.bias_add(kernel3, bias3))
	pool3 = tf.nn.max_pool(conv3, ksize=[1,3,3,1], strides=[1,2,2,1], padding='SAME')
	# 在卷积层之后使用一个全连接层
	reshape = tf.reshape(pool3, [batch_size, -1])
	dim = reshape.get_shape()[1].value
	local4 = tf.nn.relu(tf.matmul(reshape, weight4) + bias4)
	local5 = tf.nn.relu(tf.matmul(local4, weight5) + bias5)
	logits = tf.add(tf.matmul(local5, weight6), bias6) # 输出值
	logits = tf.nn.sigmoid(logits)

	sess = tf.Session()
	ini_op = tf.group(tf.global_variables_initializer(),tf.local_variables_initializer())
	sess.run(ini_op)

	labelsDict, labelsIdList, labelsNameList = loader.get_labels_attr() # 获取230个标签ID对应的30个特征
	trainDict, trainPathList, trainNameSet = loader.get_train_id() # 获取训练图片对应的标签ID

	if is_test:
		trainNameSet = loader.get_test_trainNameSet(trainPathList[39788: ], trainDict)
		wordNameList, wordAttrList, wordDict = loader.get_word_attr(trainNameSet)
		nums = 0
		num = 0
		for step in range(49028 - 39788):
			res = sess.run(logits, feed_dict={image_holder: [res_images[step]]})
			res1 = loader.count_res1(res, labelsIdList, labelsNameList)
			res2 = loader.count_res2(res1, wordNameList, wordAttrList, wordDict, trainNameSet)
			if res2[0] == label_string[step]:
				num += 1
				nums += 1
			if (step + 1) % 2000 == 0:
				print('step', step + 1, num)
				num = 0
		print('all', 49028 - 39788, nums, nums/float(49028 - 39788))
		sess.close()
	else:
		wordNameList, wordAttrList, wordDict = loader.get_word_attr(trainNameSet)
		res_images, res_names = loader.res_inputs()
		print('start count')
		res_ans = []
		for img in res_images:
			res = sess.run(logits, feed_dict={image_holder: [img]})
			res1 = loader.count_res1(res, labelsIdList, labelsNameList)
			res2 = loader.count_res2(res1, wordNameList, wordAttrList, wordDict, trainNameSet)
			res_ans.extend(res2)
		sess.close()
		return res_ans, res_names


# loader.list_to_txt(res_ans, res_names)
