#include <gtest/gtest.h>
#include "image_selection.hpp"

using namespace isaac_manipulation_rviz;

TEST(ImageSelection, MapsResizedAndLetterboxedPixels)
{
  const QSize canvas(1000, 1000), image(640, 480);
  EXPECT_EQ(imagePixel(QPoint(0, 125), canvas, image), QPoint(0, 0));
  EXPECT_EQ(imagePixel(QPoint(999, 874), canvas, image), QPoint(639, 479));
  EXPECT_EQ(imagePixel(QPoint(500, 500), canvas, image), QPoint(320, 240));
  EXPECT_EQ(imagePixel(QPoint(10, 124), canvas, image), QPoint(-1, -1));
  EXPECT_EQ(imagePixel(QPoint(10, 875), canvas, image), QPoint(-1, -1));
  EXPECT_EQ(imagePixel(QPoint(0, 0), QSize(0, 0), image), QPoint(-1, -1));
}

TEST(ImageSelection, ReadsInstanceIdsWithPaddingAndByteOrder)
{
  sensor_msgs::msg::Image mask;
  mask.width = 2;
  mask.height = 2;
  mask.step = 6;
  mask.data = {0, 0, 1, 2, 99, 99, 3, 4, 0, 0, 99, 99};
  EXPECT_EQ(instanceAt(mask, 0, 0), 0);
  EXPECT_EQ(instanceAt(mask, 1, 0), 513);
  EXPECT_EQ(instanceAt(mask, 0, 1), 1027);
  mask.is_bigendian = true;
  EXPECT_EQ(instanceAt(mask, 1, 0), 258);
  EXPECT_EQ(instanceAt(mask, 0, 1), 772);
  EXPECT_EQ(instanceAt(mask, -1, 0), 0);
  EXPECT_EQ(instanceAt(mask, 2, 0), 0);
}

TEST(ImageSelection, RequiresExactValidSourceObservation)
{
  sensor_msgs::msg::Image image, mask;
  image.width = mask.width = 2;
  image.height = mask.height = 2;
  image.header.frame_id = mask.header.frame_id = "camera";
  image.header.stamp.sec = mask.header.stamp.sec = 42;
  image.encoding = "rgb8";
  mask.encoding = "16UC1";
  image.step = 6;
  mask.step = 4;
  image.data.resize(12);
  mask.data.resize(8);
  EXPECT_TRUE(validPair(image, mask));
  mask.header.stamp.nanosec = 1;
  EXPECT_FALSE(validPair(image, mask));
  mask.header = image.header;
  mask.header.frame_id = "other";
  EXPECT_FALSE(validPair(image, mask));
  mask.header = image.header;
  mask.data.pop_back();
  EXPECT_FALSE(validPair(image, mask));
}
