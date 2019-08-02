1、训练
    在/src文件内，首先更改./core/config.py，主要需要更改的地方包括数据路径、anchor_box，学习率自行调整
	运行train_tiny.py
2、模型转换
	运行freeze_graph.py，保存pb与tflite模型。（模型的输出是score最大的box，包括：x_center,y_center,width,height，score，label。与真实图片的box位置的映射关系可参考video_demo.py）