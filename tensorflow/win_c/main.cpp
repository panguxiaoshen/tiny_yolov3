#include <stdio.h>
#include <stdlib.h>
#include <opencv2/opencv.hpp>

#include "face_detection.h"
#include "key_points.h"
#include "hand_detection.h"

using namespace cv;
using namespace std;

int main(void)
{
	/* Load face detection model from file */
	const char *model_path = "E:\\program\\keypoints_win\\test\\test\\model\\frozen_inference_graph.pb";
	if (!face_detection_init(model_path))
	{
		printf("Failed to Load model!\n");
		cv::waitKey(0);
		return -1;
	}
	printf("Load face detection model successfully!\n");

	//load face keypoints model from file
	const char *key_points_model_path = "E:\\program\\keypoints_win\\test\\test\\model\\face_keypoints.pb";
	if (!key_points_init(key_points_model_path))
	{
		printf("Failed to Load model!\n");
		cv::waitKey(0);
		return -1;
	}
	printf("Load face keypoints model successfully!\n");


	/* Load face detection model from file */
	const char *hand_model_path = "E:\\program\\keypoints_win\\test\\test\\model\\yolov3_coco.pb";
	if (!hand_detection_init(hand_model_path))
	{
		printf("Failed to Load model!\n");
		cv::waitKey(0);
		return -1;
	}
	printf("Load hand detection model successfully!\n");


	/* Face detection */
	const char *img_path2 = "E:\\program\\keypoints_win\\test\\test\\images\\2018-06-07  11.54.07_1027.jpg";
	string img_path = "E:\\program\\keypoints_win\\test\\test\\images\\*.png";
	vector<String> fn;
	glob(img_path, fn, false);
	for (int i = 0; i < fn.size(); i++)
	{
		//Mat img_mat = imread(img_path2, 0);
		Mat img_mat = imread(fn[i], 0);
		if (img_mat.empty())
		{
			cout << "Can not open the image " << img_path << endl;
			cv::waitKey(0);
			return -1;
		}

		//printf("height %d, width %d\n", img_mat.size().height, img_mat.size().width);

		/* Face detection */
		float prob = 0;
		face_box_t box;
		int ret;
		float points[52] = { 0 };

		//box.left_top_x = 194;
		//box.left_top_y = 140;
		//box.right_bottom_x = 194 + 189;
		//box.right_bottom_y = 140 + 212;

		ret = face_detection(img_mat.data, img_mat.size().height, img_mat.size().width, &box, &prob);

		if (ret != 0)
		{
			cout << "Can not detect any face~" << endl;
			continue;
			//return -1;
		}
		else
		{
		   int m = key_points_predict(img_mat.data, img_mat.size().height, img_mat.size().width, &box, points);
		   for (int i = 0; i < 26; i++)
			   circle(img_mat, Point(int(points[2 * i]), int(points[2 * i + 1])), 1, (255, 0, 0));
			   //printf("26个点的坐标按照（X，Y）依次为：%f\n", points[i]);
		}

		/* Hand detection */
		float prob_hand = 0;
		hand_box_t hand_box;
		int ret_hand;
		ret_hand = hand_detection_yolov3(img_mat.data, img_mat.size().height, img_mat.size().width, &hand_box, 160, 160);
		
		//ret_hand = hand_detection(img_mat.data, img_mat.size().height, img_mat.size().width, &hand_box, &prob_hand);

		if (ret_hand != 0)
		{
			cout << "Can not detect any hand~" << endl;
			continue;
			//return -1;
		}



		/* Debug message */
		//printf("%d %d %d %d\n", box.left_top_x, box.left_top_y, box.right_bottom_x - box.left_top_x, box.right_bottom_y - box.left_top_y);
		cv::rectangle(img_mat, Rect(box.left_top_x, box.left_top_y, box.right_bottom_x - box.left_top_x, box.right_bottom_y - box.left_top_y), Scalar(0, 0, 255), 1, 1, 0);
		cv::rectangle(img_mat, Rect(hand_box.left_top_x, hand_box.left_top_y, hand_box.right_bottom_x - hand_box.left_top_x, hand_box.right_bottom_y - hand_box.left_top_y), Scalar(0, 0, 255), 1, 1, 0);

		imshow("face detection", img_mat);
		waitKey(0);

	}

	face_detection_deinit();
	key_points_deinit();
	getchar();

	return 0;
}
