#ifndef DACINFO_H
#define DACINFO_H

#include<string>
#include<iostream>
#include<fstream>
#include<vector>

/*
	Dac 算子类，目前只含有降维算子。
*/
class Dac_Op{
	public:
		std::string name;         // 算子名称
		int split_size;           // 划分数    算子作用于不同数据划分数一致
		int split_length;         // 每份长度  算子作用于不同数据划分长度可以不同
		int dimId;                // 算子作用的维度
		std::string expression;   // 算子对应索引的计算表达式

		Dac_Op();
		/*
			通过 算子名称，算子划分数，算子作用的维度 创建算子。
		*/
		Dac_Op(std::string Name,int SplitSize,int dimId);
		/*
			设置 算子作用的维度。
		*/
		void setDimId(int id);
		/*
			设置 算子划分的每份长度。
		*/
		void setSplitLength(int len);
		/*
			设置 算子对应索引的计算表达式。
		*/
		void setExp(std::string expression);
		/*
			得到 算子对应索引的计算表达式。
		*/
		std::string getExp();
};
/*
	Dac 算子组。
	其实就是把 vector 封装了一下。
*/
class Dac_Ops{
	public:
		std::vector<Dac_Op> DacOps;
		int size;

		Dac_Ops();
		void push_back(Dac_Op x);
		Dac_Op& operator[](int i);
};
/*
	Dac 数据类，表示数据信息。
*/
class DacData{
	public:
		std::string name;                 // 数据名称
		Dac_Ops ops;                      // 被用于该数据的算子组
		int dim;                          // 数据维数
		std::vector<int> DimLength;       // 维上的长度
		std::vector<bool> isIndex;        // 该维度是否被降维
		int split_size;                   // 数据的划分数
		int split_length;                 // 数据划分的每份长度
		// int data[3][3];                   // 存放真实数据，假设类型为int用于测试数据重组

		DacData();
		/*
			通过 数据名称，数据维数，作用于各个维度的算子 创建数据。
		*/
		DacData(std::string Name,int dim,Dac_Ops ops);
		/*
			设置 数据在某维度上的长度。
		*/
		void setDimLength(int dimId,int len);
		/*
			得到 数据在某维度上的长度。
		*/
		int getDimlength(int dimId);
		/*
			设置 数据在某维度上被降维。
		*/
		void setIndex(int dimId);
		/*
			检查 数据在某维度上是否被降维。
		*/
		bool checkIndex(int dimId);
};
/*
	Dac 参数类，表示嵌入计算时传入的参数。
	其实就是封装的 DAC 数据数组。
*/
class Args{
	public:
		std::vector<DacData> args;
		int size;

		Args();
		void push_back(DacData x);
		DacData& operator[](int i);
};

#endif