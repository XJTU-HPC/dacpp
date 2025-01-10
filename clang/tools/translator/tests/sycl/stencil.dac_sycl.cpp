#include <iostream>
#include <vector>
#include <cmath>
<<<<<<< HEAD
#include "ReconTensor.h"

using namespace std;
=======
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Slice.h"
#include "/data/powerzhang/dacpp/clang/tools/translator/dacppLib/include/Tensor.hpp"


using namespace std;
using dacpp::Tensor;
>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3
namespace dacpp {
    typedef std::vector<std::any> list;
}

// 网格参数
const int NX = 32;           // x方向网格数量
const int NY = 32;           // y方向网格数量
const float Lx = 10.0f;       // x方向长度
const float Ly = 10.0f;       // y方向长度
const float alpha = 0.01f;    // 热扩散系数
const int TIME_STEPS = 1000;  // 时间步数
<<<<<<< HEAD
// 空间步长
const float dx = Lx / (NX - 1);
const float dy = Ly / (NY - 1);

// 稳定性条件
const float dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
const float delta_t = 0.4f * dt_stability; // 选择一个更严格的时间步长以确保稳定性
=======
>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3





#include <sycl/sycl.hpp>
<<<<<<< HEAD
#include "DataReconstructor.h"
#include "ParameterGeneration.h"

using namespace sycl;

void stencil(float* mat, float* out,
sycl::accessor<int, 1, sycl::access::mode::read_write> info_mat_acc,
sycl::accessor<int, 1, sycl::access::mode::read_write> info_out_acc
) 
{
    out[0] = mat[1*3+1] + alpha *delta_t * (((mat[2*3+1] - 2.0f * mat[1*3+1] + mat[0*3+1]) / (dx * dx))+ ((mat[1*3+2] - 2.0f * mat[1*3+1] + mat[1*3+0]) / (dy * dy)));
    info_mat_acc[0]=0;
    info_out_acc[0]=0;
=======
#include "/data/qinian/ice/dacpp/clang/tools/translator/dpcppLib/include/DataReconstructor.old.h"

using namespace sycl;

void stencil(float* mat, float* out) 
{
    out[0] = mat[1 * 32 + 1] + (mat[0 * 32 + 1] + mat[2 * 32 + 1] + mat[1 * 32 + 0] + mat[1 * 32 + 2] - mat[1 * 32 + 1]);
>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3
}


// 生成函数调用
<<<<<<< HEAD
void stencilShell(const dacpp::Tensor<float, 2> & matIn, dacpp::Tensor<float, 2> & matOut) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    //声明参数生成工具
    ParameterGeneration<int,2> para_gene_tool;
    // 算子初始化
    
    // 数据信息初始化
    DataInfo info_matIn;
    info_matIn.dim = matIn.getDim();
    for(int i = 0; i < info_matIn.dim; i++) info_matIn.dimLength.push_back(matIn.getShape(i));
    // 数据信息初始化
    DataInfo info_matOut;
    info_matOut.dim = matOut.getDim();
    for(int i = 0; i < info_matOut.dim; i++) info_matOut.dimLength.push_back(matOut.getShape(i));
    // 规则分区算子初始化
    RegularSlice sp1 = RegularSlice("sp1", 3, 1);
    sp1.setDimId(0);
    sp1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp1,info_matIn));

    // 规则分区算子初始化
    RegularSlice sp2 = RegularSlice("sp2", 3, 1);
    sp2.setDimId(1);
    sp2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(sp2,info_matIn));

    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.setDimId(0);
    idx1.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx1,info_matOut));

    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.setDimId(1);
    idx2.SetSplitSize(para_gene_tool.init_operetor_splitnumber(idx2,info_matOut));

    //参数生成
	
    // 参数生成 提前计算后面需要用到的参数	
	
    // 算子组初始化
    Dac_Ops matIn_Ops;
    
    sp1.setDimId(0);
    matIn_Ops.push_back(sp1);

    sp2.setDimId(1);
    matIn_Ops.push_back(sp2);


    // 算子组初始化
    Dac_Ops matOut_Ops;
    
    idx1.setDimId(0);
    matOut_Ops.push_back(idx1);

    idx2.setDimId(1);
    matOut_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops In_Ops;
    
    sp1.setDimId(0);
    In_Ops.push_back(sp1);

    sp2.setDimId(1);
    In_Ops.push_back(sp2);


    // 算子组初始化
    Dac_Ops Out_Ops;
    
    idx1.setDimId(0);
    Out_Ops.push_back(idx1);

    idx2.setDimId(1);
    Out_Ops.push_back(idx2);


    // 算子组初始化
    Dac_Ops Reduction_Ops;
    
    idx1.setDimId(0);
    Reduction_Ops.push_back(idx1);

    idx2.setDimId(1);
    Reduction_Ops.push_back(idx2);


	
    //生成设备内存分配大小
    int matIn_Size = para_gene_tool.init_device_memory_size(info_matIn,matIn_Ops);

    //生成设备内存分配大小
    int matOut_Size = para_gene_tool.init_device_memory_size(In_Ops,Out_Ops,info_matOut);

    //生成设备内存分配大小
    int Reduction_Size = para_gene_tool.init_device_memory_size(info_matOut,Reduction_Ops);

	
    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(matIn_Ops,matIn_Size);

    // 计算算子组里面的算子的划分长度
    para_gene_tool.init_op_split_length(In_Ops,matOut_Size);

	
	
    std::vector<Dac_Ops> ops_s;
	
    ops_s.push_back(matIn_Ops);

    ops_s.push_back(In_Ops);


	// 生成划分长度的二维矩阵
    int SplitLength[2][2] = {0};
    para_gene_tool.init_split_length_martix(2,2,&SplitLength[0][0],ops_s);

	
    // 计算工作项的大小
    int Item_Size = para_gene_tool.init_work_item_size(In_Ops);

	
    // 计算归约中split_size的大小
    int Reduction_Split_Size = para_gene_tool.init_reduction_split_size(In_Ops,Out_Ops);

	
    // 计算归约中split_length的大小
    int Reduction_Split_Length = para_gene_tool.init_reduction_split_length(Out_Ops);


    // 设备内存分配
    
    // 设备内存分配
    float *d_matIn=malloc_device<float>(matIn_Size,q);
    // 设备内存分配
    float *d_matOut=malloc_device<float>(matOut_Size,q);
    // 归约设备内存分配
    float *reduction_matOut = malloc_device<float>(Reduction_Size,q);
    // 数据关联计算
    
    
    // 数据重组
    DataReconstructor<float> matIn_tool;
    float* r_matIn=(float*)malloc(sizeof(float)*matIn_Size);
    
    // 数据算子组初始化
    Dac_Ops matIn_ops;
    
    sp1.setDimId(0);
    sp1.setSplitLength(8);
    matIn_ops.push_back(sp1);
    sp2.setDimId(1);
    sp2.setSplitLength(8);
    matIn_ops.push_back(sp2);
    matIn_tool.init(info_matIn,matIn_ops);
    matIn_tool.Reconstruct(r_matIn,matIn);
    std::vector<int> info_partition_matIn=para_gene_tool.init_partition_data_shape(info_matIn,matIn_ops);
    sycl::buffer<int> info_partition_matIn_buffer(info_partition_matIn.data(), sycl::range<1>(info_partition_matIn.size()));
    // 数据重组
    DataReconstructor<float> matOut_tool;
    float* r_matOut=(float*)malloc(sizeof(float)*matOut_Size);
=======
void stencilShell(const dacpp::Tensor<float> & matIn, Tensor<float> & matOut) { 
    // 设备选择
    auto selector = gpu_selector_v;
    queue q(selector);
    // 算子初始化
    
    // 规则分区算子初始化
    RegularSlice sp1 = RegularSlice("sp1", 3, 1);
    sp1.SetSplitSize(30);
    // 规则分区算子初始化
    RegularSlice sp2 = RegularSlice("sp2", 3, 1);
    sp2.SetSplitSize(30);
    // 降维算子初始化
    Index idx1 = Index("idx1");
    idx1.SetSplitSize(30);
    // 降维算子初始化
    Index idx2 = Index("idx2");
    idx2.SetSplitSize(30);
    // 数据重组
    
    // 数据重组
    DataReconstructor<float> matIn_tool;
    float* r_matIn=(float*)malloc(sizeof(float)*8100);
    
    // 数据算子组初始化
    Dac_Ops matIn_ops;

    sp1.setDimId(0);
    sp1.setSplitLength(270);
    matIn_ops.push_back(sp1);
    sp2.setDimId(1);
    sp2.setSplitLength(9);
    matIn_ops.push_back(sp2);
    
    matIn_tool.init(matIn,matIn_ops);
    matIn_tool.Reconstruct(r_matIn);
    // 数据重组
    DataReconstructor<float> matOut_tool;
    float* r_matOut=(float*)malloc(sizeof(float)*900);
>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3
    
    // 数据算子组初始化
    Dac_Ops matOut_ops;
    
    idx1.setDimId(0);
<<<<<<< HEAD
    idx1.setSplitLength(8);
    matOut_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(8);
    matOut_ops.push_back(idx2);
    matOut_tool.init(info_matOut,matOut_ops);
    matOut_tool.Reconstruct(r_matOut,matOut);
    std::vector<int> info_partition_matOut=para_gene_tool.init_partition_data_shape(info_matOut,matOut_ops);
    sycl::buffer<int> info_partition_matOut_buffer(info_partition_matOut.data(), sycl::range<1>(info_partition_matOut.size()));
    
    // 数据移动
    q.memcpy(d_matIn,r_matIn,matIn_Size*sizeof(float)).wait();
	
    //工作项划分
    sycl::range<3> local(1, 1, Item_Size);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
        auto info_partition_matIn_accessor = info_partition_matIn_buffer.get_access<sycl::access::mode::read_write>(h);
        auto info_partition_matOut_accessor = info_partition_matOut_buffer.get_access<sycl::access::mode::read_write>(h);
=======
    idx1.setSplitLength(30);
    matOut_ops.push_back(idx1);
    idx2.setDimId(1);
    idx2.setSplitLength(1);
    matOut_ops.push_back(idx2);
    matOut_tool.init(matOut,matOut_ops);
    matOut_tool.Reconstruct(r_matOut);
    // 设备内存分配
    
    // 设备内存分配
    float *d_matIn=malloc_device<float>(8100,q);
    // 设备内存分配
    float *d_matOut=malloc_device<float>(900,q);
    // 数据移动
    
    // 数据移动
    q.memcpy(d_matIn,r_matIn,8100*sizeof(float)).wait();   
    // 内核执行
    
    //工作项划分
    sycl::range<3> local(1, 1, 900);
    sycl::range<3> global(1, 1, 1);
    //队列提交命令组
    q.submit([&](handler &h) {
>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3
        h.parallel_for(sycl::nd_range<3>(global * local, local),[=](sycl::nd_item<3> item) {
            const auto item_id = item.get_local_id(2);
            // 索引初始化
			
<<<<<<< HEAD
            const auto sp1_=(item_id/sp2.split_size+(0))%sp1.split_size;
            const auto idx1_=(item_id/sp2.split_size+(0))%idx1.split_size;
            const auto sp2_=(item_id+(0))%sp2.split_size;
            const auto idx2_=(item_id+(0))%idx2.split_size;
            // 嵌入计算
            stencil(d_matIn+(sp1_*SplitLength[0][0]+sp2_*SplitLength[0][1]),d_matOut+(sp1_*SplitLength[1][0]+sp2_*SplitLength[1][1]),
                info_partition_matIn_accessor,
                info_partition_matOut_accessor
            );
        });
    }).wait();
    

	
    // 归约
    if(Reduction_Split_Size > 1)
    {
        for(int i=0;i<Reduction_Size;i++) {
            q.submit([&](handler &h) {
    	        h.parallel_for(
                range<1>(Reduction_Split_Size),
                reduction(reduction_matOut+i, 
                sycl::plus<>(),
                property::reduction::initialize_to_identity()),
                [=](id<1> idx,auto &reducer) {
                    reducer.combine(d_matOut[(i/Reduction_Split_Length)*Reduction_Split_Length*Reduction_Split_Size+i%Reduction_Split_Length+idx*Reduction_Split_Length]);
     	        });
         }).wait();
        }
        q.memcpy(d_matOut,reduction_matOut, Reduction_Size*sizeof(float)).wait();
    }


	
    // 归并结果返回
    q.memcpy(r_matOut, d_matOut, matOut_Size*sizeof(float)).wait();
    matOut_tool.UpdateData(r_matOut,matOut);

=======
            const auto sp1=(item_id/30+(0)+30)%30;
            const auto idx1=(item_id/30+( + 1)+30)%30;
            const auto sp2=(item_id+(0)+30)%30;
            const auto idx2=(item_id+( + 1)+30)%30;
            // 嵌入计算
			
            stencil(d_matIn+(sp1*270+sp2*9),d_matOut+(idx1*30+idx2*1));
        });
    }).wait();
    
    
    // 归约
    
    // 返回计算结果
    
    // 归并结果返回
    q.memcpy(r_matOut, d_matOut, 900*sizeof(float)).wait();
    matOut = matOut_tool.UpdateData(r_matOut);
>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3
    // 内存释放
    
    sycl::free(d_matIn, q);
    sycl::free(d_matOut, q);
}

int main() {
<<<<<<< HEAD

=======
    // 空间步长
    float dx = Lx / (NX - 1);
    float dy = Ly / (NY - 1);

    // 稳定性条件
    float dt_stability = (dx * dx * dy * dy) / (2.0f * alpha * (dx * dx + dy * dy));
    float delta_t = 0.4f * dt_stability; // 选择一个更严格的时间步长以确保稳定性
>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3

    cout << "Grid size: " << NX << "x" << NY << "\n";
    cout << "dx = " << dx << ", dy = " << dy << ", delta_t = " << delta_t << "\n";

    // 初始化温度场
    vector<float> u_prev(NX * NY, 0.0f); // 前一步（在热传导方程中，只有当前步和上一步）
    vector<float> u_curr(NX * NY, 0.0f);  // 当前步
    vector<float> u_next(NX * NY, 0.0f);  // 当前步

    // 初始条件：例如，中心有一个高斯分布的热源
    int cx = NX / 2;
    int cy = NY / 2;
    float sigma = 1.0f;
    for(int i = 0; i < NX; ++i) {
        for(int j = 0; j < NY; ++j) {
            float x = i * dx;
            float y = j * dy;
            // 高斯分布
            u_curr[i * NY + j] = std::exp(-((x - Lx/2.0f)*(x - Lx/2.0f) + (y - Ly/2.0f)*(y - Ly/2.0f)) / (2.0f * sigma * sigma));
        }
    }

<<<<<<< HEAD
    //std::vector<int> shape = {32, 32};
    dacpp::Tensor<float, 2> u_curr_tensor({32, 32}, u_curr);

    for(int i=0;i<TIME_STEPS;i++) {
        std::vector<float> middle_points;
=======
    for (int i = 0; i < NX; ++i) {  // 遍历每一行
        for (int j = 0; j < NY; ++j) {  // 遍历每一列
            //std::cout << u_curr[i * NY + j] << " ";  // 打印当前元素
        }
        //std::cout << std::endl;  // 每行结束后换行
    }

    std::vector<int> shape = {32, 32};
    Tensor<float> u_curr_tensor(u_curr, shape);

    for(int i=0;i<TIME_STEPS;i++) {
        std::vector<float> middle_points(NX * NY, 0.0f);
>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3
        for (int i = 1; i <= 30; i++) {
            std::vector<float> row;
            for (int j = 1; j <= 30; j++) {  
                middle_points.push_back(static_cast<float>(u_next[i*NY+j]));  
            }
            
        }
<<<<<<< HEAD
        //std::vector<int> shape2 = {30, 30};
        dacpp::Tensor<float, 2> middle_tensor({30, 30}, middle_points);
=======
        std::vector<int> shape2 = {30, 30};
        Tensor<float> middle_tensor(middle_points, shape2);
>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3

        stencilShell(u_curr_tensor, middle_tensor);

        for (int i = 1; i <= 30; i++) {
            for(int j = 1; j <=30; j++){
<<<<<<< HEAD
                float* data = new float[1];
                u_curr_tensor[i][j]=middle_tensor[i-1][j-1];
=======
                u_curr_tensor[i][j] = middle_tensor[i-1][j-1];
>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3
            }
        }

        // 处理边界条件（绝热边界：导数为零）
        for (int j = 0; j < NY; ++j) {
<<<<<<< HEAD
            //float* data = new float[1];
            u_curr_tensor[0][j]=u_curr_tensor[1][j];              // 顶部边界
            u_curr_tensor[NX - 1][j]=u_curr_tensor[NX-2][j];
             // 底部边界
        }
        for (int i = 0; i < NX; ++i) {
            //float* data = new float[1];
            u_curr_tensor[i][0]=u_curr_tensor[i][1];              // 顶部边界
            u_curr_tensor[i][NY-1]=u_curr_tensor[i][NY-2];
        }
        
    }
    u_curr_tensor[2].print();
    
=======
            u_curr_tensor[0 * NY + j] = u_curr_tensor[1 * NY + j];              // 顶部边界
            u_curr_tensor[(NX - 1) * NY + j] = u_curr_tensor[(NX - 2) * NY + j]; // 底部边界
        }
        for (int i = 0; i < NX; ++i) {
            u_curr_tensor[i * NY + 0] = u_curr_tensor[i * NY + 1];              // 左边界
            u_curr_tensor[i * NY + (NY - 1)] = u_curr_tensor[i * NY + (NY - 2)]; // 右边界
        }
        //u_curr_tensor.print();
    }
    //u_curr_tensor.print();

    float* tmp1 = new float[1024];
    u_curr_tensor.tensor2Array(tmp1);
    for (int i = 0; i < 1; i++) {
        for(int j = 0; j <NY; j++){
            std::cout<<tmp1[(i)*NX+(j)]<<" ";
        }
        std::cout<<"\n";
    }
    // float* data = new float[32 * 32];
    // u_curr_tensor.tensor2Array(data);

    // std::vector<std::vector<int>> vec2D;
    // vec2D.resize(32, std::vector<int>(32));

    // // 将一维数组的数据填充到二维数组中
    // for (int i = 0; i < 32; ++i) {
    //     for (int j = 0; j < 32; ++j) {
    //         vec2D[i][j] = data[i * 32 + j];
    //     }
    // }
    // for (const auto& row : vec2D) {  // 遍历二维vector的每一行
    //     for (int val : row) {  // 遍历每一行中的元素
    //         //std::cout << val << " ";  // 打印元素
    //     }
    //     std::cout << std::endl;  // 每行结束后换行
    // }



>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3

    // 输出最终结果的某些值作为示例
    //cout << "Final temperature at center: " << vec2D[(NX/2)*NY + (NY/2)] << "\n";

    return 0;
}
<<<<<<< HEAD
=======


>>>>>>> b51bbda3bc6f3c55673b09f047cc393151e809c3
