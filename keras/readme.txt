tiny_yolov3实现人手检测
文件说明：arm板--需要配置opencv环境，保证opencv>=4.0
		  python--人手检测python实现，保证opencv>=4.0
		  win_c--人手检测c++实现，保证opencv>=4.0
		  source:模型训练源码
		  
source说明：
1、数据准备
   	data文件夹内需包含train.txt、test.txt以及训练图片
	train.txt或者test.txt的格式为：图片绝对路径 xmin,ymin,xmax,ymax,label
		例如：path/to/img1.jpg 50,100,150,200,0

2、训练 
    python train.py
    文件相应路径以及图片的输入尺寸自行修改，调整图片输入尺寸需保证为32的倍数

3、测试
   	（1）python convert_h5_to_weights.py  -cfg_path yolov3_tiny.cfg -h5_path ./logs/trained_weights_stage_3.h5 -output_path 			  
		./model/yolov3_tiny.weights
		注意：需保证yolov3_tiny.cfg网络配置与yolov3/model.py的结构一致，否则会报错
	（2）将yolov3_tiny.cfg与yolov3_tiny.weights拷贝到对应测试坏境工程目录下进行测试即可