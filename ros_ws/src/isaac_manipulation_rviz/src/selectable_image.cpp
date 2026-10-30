#include <chrono>
#include <functional>
#include <future>
#include <map>
#include <memory>
#include <mutex>
#include <QImage>
#include <QLabel>
#include <QMouseEvent>
#include <QPainter>
#include <QVBoxLayout>
#include <QWidget>
#include <pluginlib/class_list_macros.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rviz_common/display.hpp>
#include <rviz_common/display_context.hpp>
#include <rviz_common/properties/status_property.hpp>
#include <rviz_common/properties/string_property.hpp>
#include <rviz_common/ros_integration/ros_node_abstraction_iface.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <isaac_manipulation_pointcloud_interfaces/srv/select_instance.hpp>

#include "image_selection.hpp"

namespace isaac_manipulation_rviz
{
class ImageCanvas : public QWidget
{
public:
  explicit ImageCanvas(QWidget * parent) : QWidget(parent)
  {
    setMinimumSize(160, 120);
    setCursor(Qt::PointingHandCursor);
  }
  QImage image;
  QSize painted_size;
  sensor_msgs::msg::Image::ConstSharedPtr next_mask, painted_mask;
  std::function<void(int, int)> clicked;

protected:
  void paintEvent(QPaintEvent *) override
  {
    QPainter painter(this);
    painter.fillRect(rect(), Qt::black);
    if (!image.isNull()) {
      painter.drawImage(imageRect(size(), image.size()), image);
      painted_mask = next_mask;
      painted_size = image.size();
    }
  }

  void mousePressEvent(QMouseEvent * event) override
  {
    if (event->button() != Qt::LeftButton || image.isNull()) {
      return;
    }
    const auto pixel = imagePixel(event->pos(), size(), painted_size);
    if (pixel.x() >= 0 && clicked) {
      clicked(pixel.x(), pixel.y());
    }
  }
};

class SelectableImage : public rviz_common::Display
{
  using Image = sensor_msgs::msg::Image;
  using Service = isaac_manipulation_pointcloud_interfaces::srv::SelectInstance;
  using Level = rviz_common::properties::StatusProperty;
  struct Cache
  {
    std::mutex mutex;
    std::map<int64_t, Image::ConstSharedPtr> images, masks;
  };

public:
  SelectableImage()
  {
    image_topic_ = new rviz_common::properties::StringProperty(
      "Image Topic", "/segmentation/overlay_image", "Annotated rgb8 image", this);
    mask_topic_ = new rviz_common::properties::StringProperty(
      "Mask Topic", "/segmentation/instance_mask", "Timestamp-matched 16UC1 instance mask", this);
    for (auto property : {image_topic_, mask_topic_}) {
      connect(property, &rviz_common::properties::Property::changed, this, [this]() {
        if (isEnabled() && node_) { subscribe(); }
      });
    }
  }

  ~SelectableImage() override
  {
    cancelRequest();
    delete widget_;
  }

  void onInitialize() override
  {
    node_ = context_->getRosNodeAbstraction().lock()->get_raw_node();
    client_ = node_->create_client<Service>("/pointcloud/select_instance");
    widget_ = new QWidget;
    auto layout = new QVBoxLayout(widget_);
    canvas_ = new ImageCanvas(widget_);
    label_ = new QLabel("Waiting for matching image and mask", widget_);
    label_->setWordWrap(true);
    layout->addWidget(canvas_, 1);
    layout->addWidget(label_);
    canvas_->clicked = [this](int x, int y) { select(x, y); };
    setAssociatedWidget(widget_);
  }

  void update(float, float) override
  {
    if (!isEnabled() || !cache_) { return; }
    checkResponse();
    Image::ConstSharedPtr image, mask;
    {
      std::lock_guard<std::mutex> lock(cache_->mutex);
      for (auto it = cache_->images.rbegin(); it != cache_->images.rend(); ++it) {
        auto match = cache_->masks.find(it->first);
        if (match != cache_->masks.end()) {
          image = it->second;
          mask = match->second;
          break;
        }
      }
    }
    if (!image || image == displayed_image_) { return; }
    if (!validPair(*image, *mask)) {
      report(Level::Error, "Invalid rgb8 image / 16UC1 mask pair");
      return;
    }
    canvas_->next_mask = mask;
    canvas_->image = QImage(image->data.data(), image->width, image->height,
      image->step, QImage::Format_RGB888).copy();
    displayed_image_ = image;
    canvas_->update();
    setStatus(Level::Ok, "Image", "Exact timestamp-matched image and mask; click an object to select");
  }

  void reset() override
  {
    rviz_common::Display::reset();
    if (node_ && isEnabled()) { subscribe(); }
  }

protected:
  void onEnable() override
  {
    subscribe();
    widget_->show();
  }

  void onDisable() override
  {
    image_sub_.reset();
    mask_sub_.reset();
    clearImage();
    cancelRequest();
    widget_->hide();
  }

private:
  void clearImage()
  {
    displayed_image_.reset();
    canvas_->next_mask.reset();
    canvas_->painted_mask.reset();
    canvas_->image = QImage();
    canvas_->update();
  }

  void subscribe()
  {
    cancelRequest();
    clearImage();
    cache_ = std::make_shared<Cache>();
    auto callback = [cache = cache_](bool mask, Image::ConstSharedPtr message) {
        std::lock_guard<std::mutex> lock(cache->mutex);
        auto & buffer = mask ? cache->masks : cache->images;
        const auto stamp = static_cast<int64_t>(message->header.stamp.sec) * 1000000000 +
          message->header.stamp.nanosec;
        if (!buffer.empty() && stamp < buffer.rbegin()->first) { buffer.clear(); }
        buffer[stamp] = message;
        while (buffer.size() > 10) { buffer.erase(buffer.begin()); }
      };
    try {
      image_sub_ = node_->create_subscription<Image>(image_topic_->getStdString(),
        rclcpp::SensorDataQoS().keep_last(2), [callback](Image::ConstSharedPtr msg) { callback(false, msg); });
      mask_sub_ = node_->create_subscription<Image>(mask_topic_->getStdString(),
        rclcpp::SensorDataQoS().keep_last(2), [callback](Image::ConstSharedPtr msg) { callback(true, msg); });
      report(Level::Warn, "Waiting for timestamp-matched overlay and mask; click an object when shown");
    } catch (const std::exception & error) {
      image_sub_.reset();
      mask_sub_.reset();
      report(Level::Error, QString::fromUtf8(error.what()));
    }
  }

  void select(int x, int y)
  {
    const auto mask = canvas_->painted_mask;
    if (!mask) { return; }
    const auto id = instanceAt(*mask, x, y);
    if (id == 0) { report(Level::Warn, "Background pixel; selection unchanged"); return; }
    if (pending_) { report(Level::Warn, "Selection request still pending"); return; }
    if (!client_->service_is_ready()) {
      report(Level::Error, "Selection service unavailable; start ros-perception"); return;
    }
    auto request = std::make_shared<Service::Request>();
    request->stamp = mask->header.stamp;
    request->instance_id = id;
    try {
      auto result = client_->async_send_request(request);
      future_ = result.future.share();
      request_id_ = result.request_id;
      pending_ = true;
      deadline_ = std::chrono::steady_clock::now() + std::chrono::seconds(3);
      report(Level::Warn, QString("Selecting instance %1 from displayed frame...").arg(id));
    } catch (const std::exception & error) {
      report(Level::Error, QString::fromUtf8(error.what()));
    }
  }

  void report(Level::Level level, const QString & text)
  {
    label_->setText(text);
    setStatus(level, "Selection", text);
  }

  void cancelRequest()
  {
    if (pending_) { client_->remove_pending_request(request_id_); pending_ = false; }
  }

  void checkResponse()
  {
    if (!pending_) { return; }
    if (future_.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
      pending_ = false;
      try {
        auto response = future_.get();
        report(response->success ? Level::Ok : Level::Warn,
          QString(response->success ? "Selected: " : "Rejected: ") + QString::fromStdString(response->message));
      } catch (const std::exception & error) {
        report(Level::Error, QString::fromUtf8(error.what()));
      }
    } else if (std::chrono::steady_clock::now() >= deadline_) {
      cancelRequest();
      report(Level::Error, "Selection timed out; the server may still complete the request");
    }
  }

  rviz_common::properties::StringProperty * image_topic_;
  rviz_common::properties::StringProperty * mask_topic_;
  QWidget * widget_ = nullptr;
  ImageCanvas * canvas_ = nullptr;
  QLabel * label_ = nullptr;
  rclcpp::Node::SharedPtr node_;
  rclcpp::Subscription<Image>::SharedPtr image_sub_, mask_sub_;
  rclcpp::Client<Service>::SharedPtr client_;
  std::shared_ptr<Cache> cache_;
  Image::ConstSharedPtr displayed_image_;
  std::shared_future<Service::Response::SharedPtr> future_;
  int64_t request_id_ = 0;
  bool pending_ = false;
  std::chrono::steady_clock::time_point deadline_;
};
}  // namespace isaac_manipulation_rviz

PLUGINLIB_EXPORT_CLASS(isaac_manipulation_rviz::SelectableImage, rviz_common::Display)
