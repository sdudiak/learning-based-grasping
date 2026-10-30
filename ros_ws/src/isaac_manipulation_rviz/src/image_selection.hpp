#pragma once

#include <QPoint>
#include <QRect>
#include <QSize>
#include <sensor_msgs/msg/image.hpp>

namespace isaac_manipulation_rviz
{
inline QRect imageRect(const QSize & canvas, const QSize & image)
{
  const auto scaled = image.scaled(canvas, Qt::KeepAspectRatio);
  return QRect(QPoint((canvas.width() - scaled.width()) / 2,
    (canvas.height() - scaled.height()) / 2), scaled);
}

inline QPoint imagePixel(const QPoint & point, const QSize & canvas, const QSize & image)
{
  const auto rect = imageRect(canvas, image);
  if (rect.isEmpty() || !rect.contains(point)) { return QPoint(-1, -1); }
  return QPoint((point.x() - rect.x()) * image.width() / rect.width(),
    (point.y() - rect.y()) * image.height() / rect.height());
}

inline bool validPair(const sensor_msgs::msg::Image & image, const sensor_msgs::msg::Image & mask)
{
  return image.encoding == "rgb8" && mask.encoding == "16UC1" && image.width > 0 && image.height > 0 &&
    image.width == mask.width && image.height == mask.height && image.header == mask.header &&
    image.step >= static_cast<uint64_t>(image.width) * 3 &&
    mask.step >= static_cast<uint64_t>(mask.width) * 2 &&
    image.data.size() >= static_cast<uint64_t>(image.step) * image.height &&
    mask.data.size() >= static_cast<uint64_t>(mask.step) * mask.height;
}

inline uint16_t instanceAt(const sensor_msgs::msg::Image & mask, int x, int y)
{
  if (x < 0 || y < 0 || x >= static_cast<int>(mask.width) || y >= static_cast<int>(mask.height)) {
    return 0;
  }
  const auto offset = static_cast<size_t>(y) * mask.step + static_cast<size_t>(x) * 2;
  if (offset + 1 >= mask.data.size()) { return 0; }
  return mask.is_bigendian ? (mask.data[offset] << 8) | mask.data[offset + 1] :
    mask.data[offset] | (mask.data[offset + 1] << 8);
}
}  // namespace isaac_manipulation_rviz
