#pragma once
#ifndef __HAND_DETECTION__
#define __HAND_DETECTION__

#include <tensorflow/c/c_api.h>

typedef struct
{
	int left_top_x, left_top_y, right_bottom_x, right_bottom_y;
}hand_box_t;

/*
hand detection init, create session and load model

return :
true  -- success
false -- failed
*/
bool hand_detection_init(const char *model_path);

/*
hand detection deinit
return :
-1  -- failed
0	-- success
*/
bool hand_detection_deinit(void);

/*
detection hand from the image

return:
0  : success
-1 : have no hand
return_box : location of the rectangle
prob : probability
*/
int hand_detection(unsigned char *data, int height, int width, hand_box_t *return_box, float *prob);
int hand_detection_yolov3(unsigned char *data, int height, int width, hand_box_t *return_box, int dst_height, int dst_width);

#endif