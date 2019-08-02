#include <stdio.h>
#include <stdlib.h>
#include <cstring>
#include <algorithm>
#include <tensorflow/c/c_api.h>
#include <opencv2/opencv.hpp>


static void data_deallocator(void* data, size_t len)
{
    free(data);
}

/*
	read binary pb file
		return 
			NULL 		Failed
			Otherwise	TF pb buffer(free it after create graph by yourself)
*/
TF_Buffer *read_pb_from_file(const char *pb_file_path)
{
	/* Parameter check */
	if (!pb_file_path)
	{
		printf("Please specify a pb model file!!\n");
		return NULL;
	}

	/* Read file */
	FILE *fp = fopen(pb_file_path, "rb");
	if (!fp)
	{
		printf("Failed to open pb model file %s\n", pb_file_path);
		return NULL;
	}

	/* Get file size */
	int size = 0;
	fseek(fp, 0L, SEEK_END);
	size = ftell(fp);
	fseek(fp, 0, SEEK_SET);

	/* Malloc */
	char *file_buffer = (char *)malloc(size);
	if (!file_buffer)
	{
		printf("Failed to malloc(%d)\n", size);
		fclose(fp);
		return NULL;
	}

	/* Read to buffer */
	size_t ssize = fread(file_buffer, size, 1, fp);
	if (ssize != 1)
	{
		printf("WARNING !!! fread size not equal to 1(%ld, %d)\n", ssize, size);
	}
	
	fclose(fp);

	/* Create TF buffer */	
	TF_Buffer *buf = TF_NewBuffer();
	if (!buf)
	{
		printf("Failed to Create TF buffer!\n");
		return NULL;
	}

	buf->data = file_buffer;
	buf->length = size;
	buf->data_deallocator = data_deallocator;
	
	return buf;
}

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
int load_model(const char *pb_file_path, TF_Graph **ret_graph, TF_Session **ret_session)
{
	/* Parameter check */
	if (!pb_file_path)
	{
		printf("Please specify a pb model file!!\n");
		return -1;
	}

	/* Create graph */
	TF_Graph *graph = TF_NewGraph();
	if (!graph)
	{
		printf("Failed to Create TF graph\n");
		return -1;
	}

	/* Create status */
	TF_Status* status = TF_NewStatus();
	if (!status)
	{
		TF_DeleteGraph(graph);
		printf("Failed to Create TF status\n");
		return -1;
	}

	/* Create TF options */
	TF_ImportGraphDefOptions* opts = TF_NewImportGraphDefOptions();
	if (!opts)
	{
		TF_DeleteGraph(graph);
		TF_DeleteStatus(status);
		printf("Failed to Create TF options\n");
		return -1;
	}

	/* Read pb from file */
	TF_Buffer *buffer = read_pb_from_file(pb_file_path);
	if (!buffer)
	{
		TF_DeleteGraph(graph);
		TF_DeleteStatus(status);
		TF_DeleteImportGraphDefOptions(opts);
		printf("Failed to read pb from file : %s\n", pb_file_path);
		return -1;
	}

	/* Import model to graph */
	TF_GraphImportGraphDef(graph, buffer, opts, status);
    if (TF_GetCode(status) != TF_OK)
	{
		TF_DeleteGraph(graph);
		TF_DeleteImportGraphDefOptions(opts);
		TF_DeleteBuffer(buffer);
		TF_DeleteStatus(status);		
        graph = NULL;
		printf("Failed to Import graph!\n");
		return -1;
    }

	/* Free memory */
    TF_DeleteImportGraphDefOptions(opts);
    TF_DeleteBuffer(buffer);
    TF_DeleteStatus(status);

	/* Create session */
	TF_Status* s_status = TF_NewStatus();
	TF_SessionOptions* s_options = TF_NewSessionOptions();
	TF_Session* sess = TF_NewSession(graph, s_options, s_status);
	if (TF_GetCode(s_status) != TF_OK)
	{
		TF_DeleteSessionOptions(s_options);
		TF_DeleteStatus(s_status);
		printf("Failed to create session!");
		return -1;
	}
	TF_DeleteSessionOptions(s_options);
	TF_DeleteStatus(s_status);

	*ret_graph = graph;
	*ret_session = sess;

	return 0;
}

TF_Tensor* create_tensor_and_fill_data(TF_DataType data_type, const int64_t *dims, std::size_t num_dims, cv::Mat input_mat)
{
 	if (dims == nullptr)
  	{
    	return nullptr;
  	}

	int64_t tensor_size = 1;
	for (size_t i = 0; i < num_dims; i ++)
	{
		tensor_size *= dims[i];
	}

  	TF_Tensor* tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), tensor_size);
  	if (tensor == nullptr)
	{
    	return nullptr;
  	}

  	void* tensor_data = TF_TensorData(tensor);
  	if (tensor_data == nullptr)
	{
    	TF_DeleteTensor(tensor);
    	return nullptr;
  	}

	unsigned char *p = (unsigned char *)tensor_data;

	if (dims[num_dims-1] == 3)
	{
		cv::Mat tempMat(input_mat.size().height, input_mat.size().width, CV_8UC3, p);
		input_mat.convertTo(tempMat,CV_8UC3);
	}
	else
	{
		cv::Mat tempMat(input_mat.size().height, input_mat.size().width, CV_8UC1, p);		
		input_mat.convertTo(tempMat, CV_8UC1);
	}

  	return tensor;
}

TF_Tensor* create_key_points_tensor(TF_DataType data_type, const int64_t *dims, std::size_t num_dims, cv::Mat input_mat)
{
	if (dims == nullptr)
	{
		return nullptr;
	}

	int64_t tensor_size = 1;
	for (size_t i = 0; i < num_dims; i++)
	{
		tensor_size *= dims[i];
	}

	TF_Tensor* tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), tensor_size*4);
	if (tensor == nullptr)
	{
		return nullptr;
	}

	void* tensor_data = TF_TensorData(tensor);
	if (tensor_data == nullptr)
	{
		TF_DeleteTensor(tensor);
		return nullptr;
	}

	unsigned char *p = (unsigned char *)tensor_data;

	if (dims[num_dims - 1] == 3)
	{
		cv::Mat tempMat(input_mat.size().height, input_mat.size().width, CV_8UC3, p);
		input_mat.convertTo(tempMat, CV_8UC3);
	}
	else
	{
		cv::Mat tempMat(input_mat.size().height, input_mat.size().width, CV_32FC1, p);
		tempMat = tempMat / 255.0;
		input_mat.convertTo(tempMat, CV_32FC1);
	}

	return tensor;
}

TF_Tensor* create_hand_tensor(TF_DataType data_type, const int64_t *dims, std::size_t num_dims, cv::Mat input_mat)
{
	if (dims == nullptr)
	{
		return nullptr;
	}

	int64_t tensor_size = 1;
	for (size_t i = 0; i < num_dims; i++)
	{
		tensor_size *= dims[i];
	}

	TF_Tensor* tensor = TF_AllocateTensor(data_type, dims, static_cast<int>(num_dims), tensor_size * 4);
	if (tensor == nullptr)
	{
		return nullptr;
	}

	void* tensor_data = TF_TensorData(tensor);
	if (tensor_data == nullptr)
	{
		TF_DeleteTensor(tensor);
		return nullptr;
	}

	unsigned char *p = (unsigned char *)tensor_data;

	if (dims[num_dims - 1] == 3)
	{
		cv::Mat tempMat(input_mat.size().height, input_mat.size().width, CV_8UC3, p);
		input_mat.convertTo(tempMat, CV_8UC3);
	}
	else
	{
		cv::Mat tempMat(input_mat.size().height, input_mat.size().width, CV_32FC1, p);
		tempMat = tempMat / 255.0;
		input_mat.convertTo(tempMat, CV_32FC1);
	}

	return tensor;
}




