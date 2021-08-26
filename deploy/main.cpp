#include <iostream>
#include <chrono>
#include "classification.h"
#include <fstream>
#include <string>
#include <io.h>
#include <direct.h>
#include <iostream>   
#include <WINDOWS.H>


using namespace std;


//直方图拉伸
cv::Mat contrastStretch(cv::Mat srcImage)
{
	cv::Mat resultImage = srcImage.clone();
	int nRows = resultImage.rows;
	int nCols = resultImage.cols;
	// 图像连续性判断
	if (resultImage.isContinuous())
	{
		nCols = nCols * nRows;
		nRows = 1;
	}
	// 图像指针操作
	uchar *pDataMat;
	int pixMax = 0, pixMin = 255;
	// 计算图像的最大最小值
	for (int j = 0; j < nRows; j++)
	{
		pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++)
		{
			if (pDataMat[i] > pixMax)
				pixMax = pDataMat[i];
			if (pDataMat[i] < pixMin)
				pixMin = pDataMat[i];
		}
	}
	// 对比度拉伸映射
	for (int j = 0; j < nRows; j++)
	{
		pDataMat = resultImage.ptr<uchar>(j);
		for (int i = 0; i < nCols; i++)
		{
			pDataMat[i] = (pDataMat[i] - pixMin) *
				255 / (pixMax - pixMin + 0.001) + 0.5;
		}
	}
	return resultImage;
}


/*
@in, src: 待分割的字符串
@in, delim: 分隔符字符串
@in_out, dest: 保存分割后的每个字符串
*/
void split(const string& src, const string& delim, vector<string>& dest)
{
	string str = src;
	string::size_type start = 0, index;
	string substr;

	index = str.find_first_of(delim, start);	//在str中查找(起始：start) delim的任意字符的第一次出现的位置
	while (index != string::npos)
	{
		//substr = str.substr(start, index - start);	// 保留delim之前的
		substr = str.substr(index - start +1, string::npos);// 保留delim之后的
		dest.push_back(substr);
		start = str.find_first_not_of(delim, index);	//在str中查找(起始：index) 第一个不属于delim的字符出现的位置
		if (start == string::npos) return;

		index = str.find_first_of(delim, start);
	}

}



void split2(std::string& s, std::string& delim, std::vector<std::string>* ret)
{
	size_t last = 0;
	size_t index = s.find_first_of(delim, last);
	while (index != string::npos)
	{
		ret->push_back(s.substr(last, index - last));
		last = index + 1;
		index = s.find_first_of(delim, last);
	}
	if (index - last > 0)
	{
		ret->push_back(s.substr(last, index - last));
	}
}


// move flle by read and write using std::ios::stream.
struct FileMover
{
	virtual bool move(const std::string& src, const std::string& dst) const = 0;
	bool move(const std::vector<std::string>& src, const std::vector<std::string>& dst) const;
	FileMover() {}
	virtual ~FileMover() {}
};
bool FileMover::move(const std::vector<std::string>& src, const std::vector<std::string>& dst) const {
	if (src.size() != dst.size()) {
		std::cerr << "src and dst files number not equal" << std::endl;
		return false;
	}
	bool ret = true;
	for (int i = 0; i < src.size(); ++i)
		ret &= move(src[i].c_str(), dst[i].c_str());
	return ret;
}
struct StreamMover :public FileMover {
	StreamMover(bool keep = false) :keep_src(keep) {}
	~StreamMover() {}
	bool keep_src;
	bool move(const std::string& src, const std::string& dst) const override;
};
bool StreamMover::move(const std::string& src, const std::string& dst) const {
	std::ifstream ifs(src, std::ios::binary);
	std::ofstream ofs(dst, std::ios::binary);
	if (!ifs.is_open()) {
		std::cout << "open src file fail: " + src << std::endl;
		return false;
	}
	ofs << ifs.rdbuf();
	ifs.close();
	ofs.close();
	if (!keep_src && 0 != remove(src.c_str())) {
		std::cerr << "remove src file fail: " + src << std::endl;
	}
	return true;
}
int Copy(const char *SourceFile, const char *NewFile)
{
	ifstream in;
	ofstream out;
	in.open(SourceFile, ios::binary);//打开源文件
	if (in.fail())//打开源文件失败
	{
		cout << "Error 1: Fail to open the source file." << endl;
		in.close();
		out.close();
		return 0;
	}
	out.open(NewFile, ios::binary);//创建目标文件 
	if (out.fail())//创建文件失败
	{
		cout << "Error 2: Fail to create the new file." << endl;
		out.close();
		in.close();
		return 0;
	}
	else//复制文件
	{
		out << in.rdbuf();
		out.close();
		in.close();
		return 1;
	}
}

int main()
{
	std::cout << __cplusplus << std::endl;
	
	// 设定onnx、engine、labels路径
	std::string onnxModel = "D:\\TRT_classification\\b4_best.onnx";
	std::string engine = "D:\\TRT_classification\\b4_best.engine";
	std::string label = "D:\\TRT_classification\\labels.txt";
	std::string output = "F:/traino/";
	std::string input_path = "F:\\train";
	float threshold = 0.3;
	
	// 获得所有图片路径
	cv::String folder = input_path;
	std::vector<cv::String> imagePathList;
	cv::glob(folder, imagePathList);

	// 建立输出文件夹
	ifstream infile(label, ios::in);
	vector<string> results;
	string word;
	string delim(":");
	string delim2("\\");
	string textline;
	if (infile.good())
	{
		while (!infile.fail())
		{
			getline(infile, textline);
			split(textline, delim, results);
		}
	}
	infile.close();

	vector<string>::iterator iter = results.begin();
	while (iter != results.end())
	{
		std::string prefix = output + *iter++;
		if (_access(prefix.c_str(), 0) == -1)	//如果文件夹不存在
			_mkdir(prefix.c_str());				//则创建
	}
	std::string prefix = output + "other";
	if (_access(prefix.c_str(), 0) == -1)	//如果文件夹不存在
		_mkdir(prefix.c_str());				//则创建
	

	// 初始化engine
	static ClassificationTensorRT CLASSIFICATION_TENSORRT;
	std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;
	int batch_size = 1;	//这里batchsize要和pytorch中保存onnx所用的batchsize一致
	int availableGpuNums;
	CLASSIFICATION_TENSORRT.getDevices(&availableGpuNums);
	std::cout << "Numbers of available GPU Device is " << availableGpuNums << std::endl;
		
	int time1, time2;
	time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	ctx = CLASSIFICATION_TENSORRT.classifier_initialize(onnxModel, engine, label, true, batch_size, 1, 0, 0);
	time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;
	


	// 处理每一张图片
	for (int i = 0; i < imagePathList.size(); i++) {
		std::string img_p = imagePathList[i];	// 获取图片路径
		cv::Mat img = cv::imread(img_p, cv::IMREAD_GRAYSCALE);	// 获取图片
		cv::resize(img, img, cv::Size(224, 224));	// resize图片
		cv::Mat img_his = contrastStretch(img);		// 直方图拉伸
		img_his.convertTo(img_his, CV_32FC4, 1.0 / 255, 0);		// int 2 float

		std::vector<cv::Mat>input(batch_size, img_his);		//封装成model的输入格式
		std::vector<std::vector<Prediction>>batchPredictions;	// 存放模型的输出

		// 模型推理
		time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		batchPredictions = CLASSIFICATION_TENSORRT.classifier_classify(ctx, input);
		time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
		std::cout << "Prediction Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;

		// 获得top1，top2
		const auto p = std::max_element(batchPredictions[0].begin(), 
			batchPredictions[0].end(), [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
		Prediction p2 = batchPredictions[0][1];
		
		if (p->second > threshold) {
			vector<string> tmp;
			vector<string> tmp2;
			split(p->first, delim, tmp);
			split2(img_p, delim2, &tmp2);
			string dist = output + tmp[0] +"/"+ tmp2[tmp2.size()-1];
			Copy(img_p.c_str(), dist.c_str());
			//StreamMover sm;
			//sm.move(img_p, dist);
		}
		else {
			vector<string> tmp;
			vector<string> tmp2;
			split(p->first, delim, tmp);
			split2(img_p, delim2, &tmp2);
			string dist = output + "other" + "/" + tmp2[tmp2.size() - 1];
			Copy(img_p.c_str(), dist.c_str());
		}
	

	}
}


	//std::cout << std::endl;
	//time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//std::cout << "Classification Time : " << time2 - time1 << "ms" << std::endl;


    //// 1、get one image
	//cv::Mat img = cv::imread("D:\\TRT_classification\\2.bmp",cv::IMREAD_GRAYSCALE);
	//std::string onnxModel = "D:\\TRT_classification\\b4_best.onnx";
	////std::string label = "D:\\TRT_classification\\labels.txt";
	//std::string label = "";
	//std::string engine = "D:\\TRT_classification\\b4_best.engine";
	//std::shared_ptr<ClassificationTensorRT::classifier_ctx> ctx;
	//	
	//int time1, time2;
	////这里batchsize要和pytorch中保存onnx所用的batchsize一致
	//int batch_size = 1;
	//int availableGpuNums;
	//CLASSIFICATION_TENSORRT.getDevices(&availableGpuNums);
	//std::cout << "Numbers of available GPU Device is " << availableGpuNums << std::endl;
	//	
	//time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//ctx = CLASSIFICATION_TENSORRT.classifier_initialize(onnxModel, engine, label, true, batch_size, 1, 0, 0);
	//time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//std::cout << "Initialization Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;


	////cv::resize(img, img, cv::Size(150,150));
	//cv::resize(img, img, cv::Size(224, 224));
	//
	//cv::Mat img_his = contrastStretch(img);
	//img_his.convertTo(img_his, CV_32FC4, 1.0 / 255, 0);

	//std::vector<cv::Mat>input(batch_size, img_his);
	//std::vector<std::vector<Prediction>>batchPredictions;
	//	

	//time1 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//batchPredictions = CLASSIFICATION_TENSORRT.classifier_classify(ctx, input);
	//time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//std::cout << "Prediction Time : " << (time2 - time1) / 1000.0 << "s" << std::endl;
	//	
	//	
	//const auto p = std::max_element(batchPredictions[0].begin(), batchPredictions[0].end(), [](const auto& lhs, const auto& rhs) { return lhs.second < rhs.second; });
	//std::cout << "\nTOP 1 Prediction \n" << p->first << " : " << std::to_string(p->second * 100) << "%\n" << std::endl;
	//std::cout << "\nTOP 5 Predictions\n";
	//for (size_t i = 0; i < batchPredictions[0].size(); ++i)
	//{
	//	Prediction p = batchPredictions[0][i];
	//	std::cout << p.first << " : " << std::to_string(p.second * 100) << "%\n";
	//}
	//	
	//std::cout << std::endl;
	//time2 = std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::steady_clock::now().time_since_epoch()).count();
	//std::cout << "Classification Time : " << time2 - time1 << "ms" << std::endl;

