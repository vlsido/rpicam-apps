#include "opencv2/face/facerec.hpp"
#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <libcamera/stream.h>
#include <memory>
#include <opencv2/core/types.hpp>
#include <string>
#include <vector>

#include <libcamera/geometry.h>

#include "core/completed_request.hpp"
#include "core/rpicam_app.hpp"
#include "post_processing_stages/post_processing_stage.hpp"

#include "opencv2/face.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/objdetect.hpp"

using namespace cv;
using namespace cv::face;

using Stream = libcamera::Stream;

class FaceRecognitionCvStage : public PostProcessingStage
{
public:
	FaceRecognitionCvStage(RPiCamApp *app) : PostProcessingStage(app) {}

	char const *Name() const override;

	void Read(boost::property_tree::ptree const &params) override;

	void Configure() override;

	bool Process(CompletedRequestPtr &completed_request) override;

	void Stop() override;

private:
	void detectAndRecognizeFeatures(cv::CascadeClassifier &cascade);
	void drawFeaturesAndNames(cv::Mat &img);

	struct RecognizedFace
	{
		Rect rect;
		std::string label;
		double confidence;
	};

	Stream *stream_;
	StreamInfo low_res_info_;
	Stream *full_stream_;
	StreamInfo full_stream_info_;
	std::unique_ptr<std::future<void>> future_ptr_;
	std::mutex face_mutex_;
	std::mutex future_ptr_mutex_;
	Mat image_;
	std::vector<RecognizedFace> recognized_faces_;
	std::vector<cv::Rect> faces_;
	CascadeClassifier cascade_;
	std::string last_recognized_face_;
	std::chrono::milliseconds last_recognized_face_timestamp_;
	Ptr<LBPHFaceRecognizer> recognizer_;
	std::string cascadeName_;
	std::string embeddings_file_;
	std::string labels_file_;
	std::vector<std::string> labels_;
	double scaling_factor_;
	int min_neighbors_;
	int min_size_;
	int max_size_;
	int refresh_rate_;
	int draw_features_;
	double confidence_threshold_;
};

#define NAME "face_recognition_cv"

char const *FaceRecognitionCvStage::Name() const
{
	return NAME;
}

void FaceRecognitionCvStage::Read(boost::property_tree::ptree const &params)
{
	cascadeName_ =
		params.get<char>("cascade_name", "/usr/local/share/opencv4/haarcascades/haarcascade_frontalface_alt.xml");
	if (!cascade_.load(cascadeName_))
	{
		throw std::runtime_error("Failed to load haar classifier");
	}
	embeddings_file_ = params.get<std::string>("embeddings_file", "embeddings.xml");
	labels_file_ = params.get<std::string>("labels_file", "labels.txt");
	scaling_factor_ = params.get<double>("scaling_factor", 1.1);
	min_neighbors_ = params.get<int>("min_neighbors", 3);
	min_size_ = params.get<int>("min_size", 32);
	max_size_ = params.get<int>("max_size", 256);
	refresh_rate_ = params.get<int>("refresh_rate", 1);
	draw_features_ = params.get<int>("draw_features", 0);
	confidence_threshold_ = params.get<double>("confidence_threshold", 50.0);
}

void FaceRecognitionCvStage::Configure()
{
	stream_ = nullptr;
	full_stream_ = nullptr;

	if (app_->StillStream()) // If still stream, no more configuration needed
		return;

	stream_ = app_->LoresStream();
	if (!stream_)
		throw std::runtime_error("FaceRecognitionCvStage No lores stream");

	// the low res stream can only be YUV420
	low_res_info_ = app_->GetStreamInfo(stream_);
	// We also expect there to be a "full resolution" stream which defines the
	// output coordinate system, and we can optionally draw the faces there too.
	full_stream_ = app_->GetMainStream();
	if (!full_stream_)
		throw std::runtime_error("FaceDetectCvStage: no full resolution stream available");
	full_stream_info_ = app_->GetStreamInfo(full_stream_);
	if (draw_features_ && full_stream_->configuration().pixelFormat != libcamera::formats::YUV420)
		throw std::runtime_error("FaceDetectCvStage: drawing only supported for YUV420 images");

	recognizer_ = LBPHFaceRecognizer::create();
	recognizer_->read(embeddings_file_);

	std::ifstream labelFile(labels_file_);
	std::string line;
	while (std::getline(labelFile, line))
	{
		labels_.push_back(line);
	}
}

bool FaceRecognitionCvStage::Process(CompletedRequestPtr &completed_request)
{
	if (!stream_)
		return false;

	{
		std::unique_lock<std::mutex> lck(future_ptr_mutex_);
		if (completed_request->sequence % refresh_rate_ == 0 &&
			(!future_ptr_ || future_ptr_->wait_for(std::chrono::seconds(0)) == std::future_status::ready))
		{
			BufferReadSync r(app_, completed_request->buffers[stream_]);
			libcamera::Span<uint8_t> buffer = r.Get()[0];
			uint8_t *ptr = (uint8_t *)buffer.data();
			Mat image(low_res_info_.height, low_res_info_.width, CV_8U, ptr, low_res_info_.stride);
			image_ = image.clone();

			future_ptr_ = std::make_unique<std::future<void>>();
			*future_ptr_ = std::async(std::launch::async, [this] { detectAndRecognizeFeatures(cascade_); });
		}
	}

	std::unique_lock<std::mutex> lock(face_mutex_);

	std::vector<libcamera::Rectangle> temprect;
	std::transform(faces_.begin(), faces_.end(), std::back_inserter(temprect),
				   [](Rect &r) { return libcamera::Rectangle(r.x, r.y, r.width, r.height); });
	completed_request->post_process_metadata.Set("detected_faces", temprect);

	completed_request->post_process_metadata.Set("last_recognized_face", last_recognized_face_);

	completed_request->post_process_metadata.Set("last_recognized_face_timestamp", last_recognized_face_timestamp_);
	if (draw_features_)
	{
		BufferWriteSync w(app_, completed_request->buffers[full_stream_]);
		libcamera::Span<uint8_t> buffer = w.Get()[0];
		uint8_t *ptr = (uint8_t *)buffer.data();
		Mat image(full_stream_info_.height, full_stream_info_.width, CV_8U, ptr, full_stream_info_.stride);
		drawFeaturesAndNames(image);
	}

	return false;
}

void FaceRecognitionCvStage::detectAndRecognizeFeatures(CascadeClassifier &cascade)
{
	equalizeHist(image_, image_);

	std::vector<Rect> temp_faces;
	cascade.detectMultiScale(image_, temp_faces, scaling_factor_, min_neighbors_, CASCADE_SCALE_IMAGE,
							 Size(min_size_, min_size_), Size(max_size_, max_size_));

	// Scale faces back to the size and location in the full res image.
	double scale_x = full_stream_info_.width / (double)low_res_info_.width;
	double scale_y = full_stream_info_.height / (double)low_res_info_.height;

	std::vector<RecognizedFace> temp_recognized_faces;
	for (auto &face : temp_faces)
	{
		Mat face_roi = image_(face);
		int label;
		double confidence;
		recognizer_->predict(face_roi, label, confidence);

		RecognizedFace recognized_face;
		recognized_face.rect = Rect(face.x * scale_x, face.y * scale_y, face.width * scale_x, face.height * scale_y);
		recognized_face.label = (confidence < confidence_threshold_) ? labels_[label] : "Unknown";
		recognized_face.confidence = confidence;

		temp_recognized_faces.push_back(recognized_face);

		if (recognized_face.label != "Unknown")
		{
			last_recognized_face_ = recognized_face.label;
			last_recognized_face_timestamp_ = std::chrono::duration_cast<std::chrono::milliseconds>(
				std::chrono::system_clock::now().time_since_epoch());
		}
	}
	std::unique_lock<std::mutex> lock(face_mutex_);
	recognized_faces_ = std::move(temp_recognized_faces);
}

void FaceRecognitionCvStage::drawFeaturesAndNames(Mat &img)
{
	for (const auto &face : recognized_faces_)
	{
		rectangle(img, face.rect, Scalar(255, 0, 0), 2);

		std::string label = face.label + " (" + std::to_string(int(100 - face.confidence)) + "%)";
		int baseline = 0;
		Size text_size = getTextSize(label, FONT_HERSHEY_COMPLEX, 4, 4, &baseline);
		Point text_org(face.rect.x, face.rect.y - text_size.height - 5);

		rectangle(img, text_org + Point(0, baseline), text_org + Point(text_size.width, -text_size.height),
				  Scalar(255, 0, 255), -1);
		putText(img, label, text_org, FONT_HERSHEY_COMPLEX, 4, Scalar(30, 100, 255), 5);
	}
}

void FaceRecognitionCvStage::Stop()
{
	if (future_ptr_)
		future_ptr_->wait();
}

static PostProcessingStage *Create(RPiCamApp *app)
{
	return new FaceRecognitionCvStage(app);
}

static RegisterStage reg(NAME, &Create);
