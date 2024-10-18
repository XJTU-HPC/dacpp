#include "sub_template.h"
int main(){
	std::cout<<"************************************dac2sycl blockMatMul CodeGen test************************************\n\n";
	
    // 设备内存分配
    std::string deviceMemAlloc = CodeGen_DeviceMemAlloc("int","matA", "16");
    deviceMemAlloc += CodeGen_DeviceMemAlloc("int","matB", "16");
    deviceMemAlloc += CodeGen_DeviceMemAlloc("int","matC", "16*2");
    std::cout<<deviceMemAlloc;
    
    // 主机数据移动至设备
    std::string H2DMemMove = CodeGen_H2DMemMov("int","matA", "16");
    H2DMemMove += CodeGen_H2DMemMov("int","matB","16");


    // 索引初始化 得到在工作项中的索引值 需要算子的步长stride与size
	// 为了统一，默认降维算子的stride与size均为1
	RegularSlice s1 = RegularSlice("s1", 2, 2);
	s1.SetSplitSize(2);
	RegularSlice s2 = RegularSlice("s2", 2, 2);
	s2.SetSplitSize(2);
	Dac_Ops ops;
	ops.push_back(s1);
	ops.push_back(s2);
	std::string IndexInit1 = CodeGen_IndexInit(ops);

	RegularSlice s3 = RegularSlice("s3", 2, 2);
	s3.SetSplitSize(2);
	ops.push_back(s3);
	std::string IndexInit2 = CodeGen_IndexInit(ops);

	Index i = Index("i");
	i.SetSplitSize(2);
	Index j = Index("j");
	j.SetSplitSize(2);
	ops.push_back(i);
	ops.push_back(j);
	std::string IndexInit3 = CodeGen_IndexInit(ops);


	// 嵌入计算 需要算子的作用维度 算子的每份划分长度
	Dac_Ops matA_ops;
	s1.setDimId(0);
	s1.setSplitLength(8);
	matA_ops.push_back(s1);
	s3.setDimId(1);
	s3.setSplitLength(4);
	matA_ops.push_back(s3);
	i.setDimId(0);
	i.setSplitLength(2);
	matA_ops.push_back(i);
	DacData d_matA = DacData("d_matA", 2, matA_ops);

	Dac_Ops matB_ops;
	s2.setDimId(1);
	s2.setSplitLength(8);
	matB_ops.push_back(s2);
	s3.setDimId(0);
	s3.setSplitLength(4);
	matA_ops.push_back(s3);
	j.setDimId(1);
	j.setSplitLength(2);
	matB_ops.push_back(j);
	DacData d_matB = DacData("d_matB", 2, matB_ops);

	Dac_Ops matC_ops;
	s1.setDimId(0);
	s1.setSplitLength(16);
	matC_ops.push_back(s1);
	s2.setDimId(1);
	s2.setSplitLength(8);
	matC_ops.push_back(s2);
	s3.setDimId(2);
	s3.setSplitLength(4);
	matC_ops.push_back(s3);
	i.setDimId(0);
	i.setSplitLength(2);
	matC_ops.push_back(i);
	j.setDimId(1);
	j.setSplitLength(2);
	matC_ops.push_back(j);
	DacData d_matC = DacData("d_matC", 3, matC_ops);

	Args args = Args();
	args.push_back(d_matA);
	args.push_back(d_matB);
	args.push_back(d_matC);

	std::string CalcEmbed = CodeGen_CalcEmbed("block_mat_mul",args);
	std::string KernelExecute = CodeGen_KernelExecute("32",IndexInit3,CalcEmbed);
	std::cout<<KernelExecute;
}