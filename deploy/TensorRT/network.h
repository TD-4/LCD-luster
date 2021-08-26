#pragma once


enum class NetWorkType : int
{
	kLUSC = 0, //!<  自研分类网络
	kLUSS = 1,  //!< 自研分割网络
	kLUSD = 2,  //!< 自研检测网络
	kRES50 = 3, //!<  ResNet50
	kLED = 4   //!<   LEDNet
};