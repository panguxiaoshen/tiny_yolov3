#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>

/*
	Load your model
	Parameters
		pb_file_path	--		your model file path
		ret_graph		--		return created graph
		ret_session		--		return created session
	Return
		0	--	success
		-1	--	failed
*/
int load_model(const char *pb_file_path, TF_Graph **ret_graph, TF_Session **ret_session);

/*
	Create tensor and fill data
		Parameters
			data_type	--	tensor data type
			dims		--  tensor dimensions(such as {1,300,300,1})
			num_dims	--	tensor dimensions len(such as 4)
			data		--  data to fill into tensor(such as data from mat.data)
			data_len	--  data length
*/
TF_Tensor* create_tensor_and_fill_data(TF_DataType data_type, const int64_t *dims, std::size_t num_dims, cv::Mat input_mat);

TF_Tensor* create_key_points_tensor(TF_DataType data_type, const int64_t *dims, std::size_t num_dims, cv::Mat input_mat);
TF_Tensor* create_hand_tensor(TF_DataType data_type, const int64_t *dims, std::size_t num_dims, cv::Mat input_mat);

