
#include <iostream>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>

#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>

#include "utils.h"
#include "face_detection.h"
#include "hand_detection.h"

using namespace std;
using namespace cv;

/* Graph and Session */
static TF_Graph *s_graph_hand_dectection = NULL;
static TF_Session *s_session_hand_dectection = NULL;
/*
TF_Output GetOperationByName(std::string name, int idx)
{
	TF_Operation *op = TF_GraphOperationByName(s_graph_hand_dectection, name.c_str());

	if (!op)
	{
		printf("Failed to get operation (%s)\n", name.c_str());
	}

	return{ op, idx };
}
*/
/*
hand detection init, create session and load model

return :
-1  -- failed
0	-- success
*/
bool hand_detection_init(const char *model_path)
{
	return load_model(model_path, &s_graph_hand_dectection, &s_session_hand_dectection) ? false : true;
}

/*
hand detection deinit
return :
-1  -- failed
0	-- success
*/
bool hand_detection_deinit(void)
{
	TF_Status* status = TF_NewStatus();

	/* Close session */
	TF_CloseSession(s_session_hand_dectection, status);
	if (TF_GetCode(status) != TF_OK)
	{
		printf("Error close session\n");
		TF_DeleteStatus(status);
		return false;
	}

	/* Delete session */
	TF_DeleteSession(s_session_hand_dectection, status);
	if (TF_GetCode(status) != TF_OK)
	{
		printf("Error delete session");
		TF_DeleteStatus(status);
		return false;
	}

	/* Delete graph */
	TF_DeleteGraph(s_graph_hand_dectection);

	return true;
}

hand_box_t hand_detection_get_true_box(hand_box_t box, int padding_len, bool padding_left_right)
{

	hand_box_t ret_box;

	if (!padding_left_right)
	{
		/* fixed x */
		ret_box.left_top_x = box.left_top_x;
		ret_box.right_bottom_x = box.right_bottom_x;

		/* modify y */
		ret_box.left_top_y = box.left_top_y - padding_len;
		ret_box.right_bottom_y = box.right_bottom_y - padding_len;
	}
	else
	{
		/* fixed y */
		ret_box.left_top_y = box.left_top_y;
		ret_box.right_bottom_y = box.right_bottom_y;

		/* modify x */
		ret_box.left_top_x = box.left_top_x - padding_len;
		ret_box.right_bottom_x = box.right_bottom_x - padding_len;
	}

	return ret_box;
}
#if 0
int get_true_hand_box_yolov3(int src_width, int src_height, float* src_box, int input_size, hand_box_t*dst_box)
{
	float xmin = src_box[0] - src_box[2] / 2;
	float xmax = src_box[0] + src_box[2] / 2;
	float ymin = src_box[1] - src_box[3] / 2;
	float ymax = src_box[1] + src_box[3] / 2;
	//LOGD("*** hand_detect score1:%f,%f,%f,%f",  xmin,xmax,ymin,ymax);
	float resize_ratio = 0;
	if (input_size / src_width <= input_size / src_height)
	{
		resize_ratio = input_size / (float)src_width;
	}
	else
	{
		resize_ratio = input_size / (float)src_height;
	}
	float dw = (input_size - resize_ratio * src_width) / 2;
	float dh = (input_size - resize_ratio * src_height) / 2;
	//LOGD("*** hand_detect score2:%f,%f",  dw,dh);
	int new_xmin = (int)((xmin - dw) / resize_ratio);
	int new_xmax = (int)((xmax - dw) / resize_ratio);
	int new_ymin = (int)((ymin - dh) / resize_ratio);
	int new_ymax = (int)((ymax - dh) / resize_ratio);
	dst_box->left_top_x = new_xmin;
	dst_box->left_top_y = new_ymin;
	dst_box->right_bottom_x = new_xmax;
	dst_box->right_bottom_y = new_ymax;
	if (dst_box->left_top_x<0) dst_box->left_top_x = 0;
	if (dst_box->left_top_y<0) dst_box->left_top_y = 0;
	if (dst_box->right_bottom_x>src_width) dst_box->right_bottom_x = src_width;
	if (dst_box->right_bottom_y>src_height) dst_box->right_bottom_y = src_height;
	// LOGD("*** hand_detect score:%f,%d,%d",  resize_ratio,src_width,src_height);
	//LOGD("*** hand_detect score3:%d,%d,%d,%d",  dst_box[0],dst_box[1],dst_box[2],dst_box[3]);
	return 0;
}
#endif
/*
detection hand from the image

return
<0 -- Failed
0 -- Success
*/
int hand_detection_yolov3(unsigned char *data, int src_height, int src_width, hand_box_t *return_box, int dst_height, int dst_width)
{
	cv::Mat input_mat(src_height, src_width, CV_8UC1, data);
	
	int padding_len;
	bool padding_left_right;
	float scale=0;
	if (src_height > src_width)
	{
		scale = dst_height / (float)src_height;
		cv::resize(input_mat, input_mat, Size((int)scale*src_width, (int)scale*src_height));
		int img_width = input_mat.size().width;
		int img_height = input_mat.size().height;

		padding_len = (img_height - img_width) / 2;
		cv::copyMakeBorder(input_mat, input_mat, 0, 0, padding_len, padding_len, cv::BORDER_CONSTANT);
		img_width = img_height;
		padding_left_right = true;

	}
	else
	{
		scale = dst_width / (float)src_width;
		cv::resize(input_mat, input_mat, Size((int)(scale*src_width), (int)(scale*src_height)));
		
		int img_width = input_mat.size().width;
		int img_height = input_mat.size().height;
		//cout << img_width << img_height << endl;
		padding_len = (img_width - img_height) / 2;
		cv::copyMakeBorder(input_mat, input_mat, padding_len, padding_len, 0, 0, cv::BORDER_CONSTANT);
		img_height = img_width;
		padding_left_right = false;
		

	}
	
	cv::resize(input_mat,input_mat,cv::Size(dst_width,dst_height));
	Mat input;
	input_mat.convertTo(input, CV_32FC1, 1 / 255.0);
	//imshow("face detection", input_mat);
	//waitKey(0);
	//input_mat.convertTo(input_mat, CV_32FC1, 1 / 255.0);
#if 0	
	for (int i = 0; i < 160; i++)
	{
		for (int j = 0; j < 160; j++)
		{
			cout << input_mat.ptr<float>(i)[j] << endl; 
		}
	}
#endif
	/* Create input tensor and fill input data(remember to free the input_tensor after use) */
	const std::vector<int64_t> input_dims = { 1, dst_width, dst_height, 1 };
	
	TF_Tensor* input_tensor = create_hand_tensor(TF_FLOAT, input_dims.data(), input_dims.size(), input);
	TF_Output input_op = { TF_GraphOperationByName(s_graph_hand_dectection, "input/input_data"), 0 };
	if (input_op.oper == nullptr)
	{
		printf("Can't init image_tensor\n");
		return -1;
	}

	/* Get output operations */
	std::vector<TF_Output> out_op = { { TF_GraphOperationByName(s_graph_hand_dectection, "output"), 0 } };
	/* Session run */
	std::vector<TF_Tensor*> output_tensors = { nullptr };
	TF_Status* status = TF_NewStatus();
	TF_SessionRun(s_session_hand_dectection,
		nullptr, // Run options.
		&input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
		out_op.data(), output_tensors.data(), 1, // Output tensors, output tensor values, number of outputs.
		nullptr, 0, // Target operations, number of targets.
		nullptr, // Run metadata.
		status // Output status.
		);

	/* Result check */
	if (TF_GetCode(status) != TF_OK)
	{
		printf("Error run session!\n");
		TF_DeleteStatus(status);
		return -1;
	}

	/* Get reulst:score and box */
	const auto out_box = static_cast<float*>(TF_TensorData(output_tensors[0]));
	int true_box[4] = { 0 };
	if (out_box[4] < 0.7)
	{
		TF_DeleteTensor(input_tensor);
		TF_DeleteTensor(output_tensors[0]);
		TF_DeleteStatus(status);
		return -1;
	}
	return_box->left_top_x		= out_box[0] - out_box[2] / 2;
	return_box->right_bottom_x  = out_box[0] + out_box[2] / 2;
	return_box->left_top_y		= out_box[1] - out_box[3] / 2;
	return_box->right_bottom_y	= out_box[1] + out_box[3] / 2;
	cout << padding_len << endl;
	//cout << return_box << endl;


#if 0
	cv::rectangle(input_mat, Rect(return_box->left_top_x, return_box->left_top_y, return_box->right_bottom_x - return_box->left_top_x,
		return_box->right_bottom_y - return_box->left_top_y), Scalar(0, 0, 255), 1, 1, 0);
	imshow("face detection", input_mat);
	waitKey(0);
#endif
	(*return_box) = hand_detection_get_true_box(*return_box, padding_len, padding_left_right);
	return_box->left_top_x /= scale;
	return_box->right_bottom_x /= scale;
	return_box->left_top_y /= scale;
	return_box->right_bottom_y /= scale;
	
	

	TF_DeleteTensor(input_tensor);
	TF_DeleteTensor(output_tensors[0]);
	TF_DeleteStatus(status);

	return 0;

}
int hand_detection(unsigned char *data, int height, int width, hand_box_t *return_box, float *prob)
{
	/* Data into cv MAT */
	cv::Mat mat(height, width, CV_8UC1, data);
	cv::Mat img_mat;

	cvtColor(mat, img_mat, COLOR_GRAY2RGB);

	/* Padding image */
	cv::Mat input_mat;
	int img_width = img_mat.size().width;
	int img_height = img_mat.size().height;

	bool padding_left_right;
	int padding_len;
	if (img_height > img_width)
	{
		padding_len = (img_height - img_width) / 2;
		cv::copyMakeBorder(img_mat, input_mat, 0, 0, padding_len, padding_len, cv::BORDER_CONSTANT);
		img_width = img_height;
		padding_left_right = true;
	}
	else
	{
		padding_len = (img_width - img_height) / 2;
		cv::copyMakeBorder(img_mat, input_mat, padding_len, padding_len, 0, 0, cv::BORDER_CONSTANT);
		img_height = img_width;
		padding_left_right = false;
	}

	/* Create input tensor and fill input data(remember to free the input_tensor after use) */
	const std::vector<int64_t> input_dims = { 1, img_height, img_width, 3 };
	TF_Tensor* input_tensor = create_tensor_and_fill_data(TF_UINT8, input_dims.data(), input_dims.size(), input_mat);


#if 0
	/* Tensor debug message */
	printf("Tensor data type %d, num dims %d, (%ld, %ld, %ld, %ld), byte size %ld\n", TF_TensorType(input_tensor), TF_NumDims(input_tensor),
		TF_Dim(input_tensor, 0), TF_Dim(input_tensor, 1), TF_Dim(input_tensor, 2), TF_Dim(input_tensor, 3), TF_TensorByteSize(input_tensor));
#endif
	/* Get input operations */
	TF_Output input_op = { TF_GraphOperationByName(s_graph_hand_dectection, "image_tensor"), 0 };
	if (input_op.oper == nullptr)
	{
		printf("Can't init image_tensor\n");
		return -1;
	}

	/* Get output operations */
	std::vector<TF_Output> out_op = { { TF_GraphOperationByName(s_graph_hand_dectection, "detection_scores"), 0 }, { TF_GraphOperationByName(s_graph_hand_dectection, "detection_boxes"), 0 } };

	/* Session run */
	std::vector<TF_Tensor*> output_tensors = { nullptr, nullptr };
	TF_Status* status = TF_NewStatus();
	TF_SessionRun(s_session_hand_dectection,
		nullptr, // Run options.
		&input_op, &input_tensor, 1, // Input tensors, input tensor values, number of inputs.
		out_op.data(), output_tensors.data(), 2, // Output tensors, output tensor values, number of outputs.
		nullptr, 0, // Target operations, number of targets.
		nullptr, // Run metadata.
		status // Output status.
		);

	/* Result check */
	if (TF_GetCode(status) != TF_OK)
	{
		printf("Error run session!\n");
		TF_DeleteStatus(status);
		return -1;
	}

	/* Get reulst:score and box */
	const auto out_score = static_cast<float*>(TF_TensorData(output_tensors[0]));
	printf("Probability %f\n", out_score[0]);

	const auto out_box = static_cast<float*>(TF_TensorData(output_tensors[1]));
	printf("Box (%f, %f, %f, %f)\n", out_box[0], out_box[1], out_box[2], out_box[3]);

	if (out_score[0] < 0.6)
	{
		TF_DeleteTensor(input_tensor);
		TF_DeleteTensor(output_tensors[0]);
		TF_DeleteTensor(output_tensors[1]);
		TF_DeleteStatus(status);
		return -1;
	}

	if (prob)
	{
		*prob = out_score[0];
	}

	return_box->left_top_x = (int)(out_box[1] * img_width);
	return_box->left_top_y = (int)(out_box[0] * img_height);
	return_box->right_bottom_x = (int)(out_box[3] * img_width);
	return_box->right_bottom_y = (int)(out_box[2] * img_height);

	(*return_box) = hand_detection_get_true_box(*return_box, padding_len, padding_left_right);

	TF_DeleteTensor(input_tensor);
	TF_DeleteTensor(output_tensors[0]);
	TF_DeleteTensor(output_tensors[1]);
	TF_DeleteStatus(status);

	return 0;
}





