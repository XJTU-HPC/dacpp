#include "sub_template.h"

int main(){
	std::cout<<"************************************dac2sycl bindIndexInit CodeGen test************************************\n\n";
    Index i = Index("i");
	i.SetSplitSize(3);
	Index j = Index("j");
	j.SetSplitSize(4);
    Index k = Index("k");
    k.SetSplitSize(5);
    Index l = Index("l");
	l.SetSplitSize(3);
	
	Dac_Ops ops;
	ops.push_back(i);
	ops.push_back(j);
    ops.push_back(k);
    ops.push_back(l);

    std::vector<std::string> sets = {"id0", "id1", "id2", "id0"};
    std::vector<int> offsets = {0, 1, 2, 3};
	// std::vector<std::string> sets = {"id0","id1"};
	// std::vector<int> offsets = {0,1};
	std::string IndexInit = CodeGen_IndexInit(ops,sets,offsets);

	std::cout<<IndexInit;
	return 0;

	// Dac_Ops vecA_ops;
	// i.setDimId(0);
	// i.setSplitLength(1);
	// vecA_ops.push_back(i);
	// DacData d_vecA = DacData("d_vecA",1,vecA_ops);
	// // d_vecA.setDimLength(0,3);

	// Dac_Ops vecB_ops;
	// j.setDimId(0);
	// j.setSplitLength(1);
	// vecB_ops.push_back(j);
	// DacData d_vecB = DacData("d_vecB",1,vecB_ops);
	// // d_vecB.setDimLength(0,3);

	// Dac_Ops dotProduct_ops;
	// i.setSplitLength(3);
	// j.setSplitLength(1);
	// j.setDimId(1);
	// dotProduct_ops.push_back(i);
	// dotProduct_ops.push_back(j);
	// DacData d_dotProduct = DacData("d_dotProduct",2,dotProduct_ops);
	// // d_dotProduct.setDimLength(0,3);
	// // d_dotProduct.setDimLength(1,3);

	// Args args = Args();
	// args.push_back(d_vecA);
	// args.push_back(d_vecB);
	// args.push_back(d_dotProduct);
	// std::string CalcEmbed = CodeGen_CalcEmbed("mat_mul",args);

	// std::string KernelExecute = CodeGen_KernelExecute("9",IndexInit,CalcEmbed);

}