#include "../NiuTensor/source/tensor/XTensor.h"
#include "../NiuTensor/source/tensor/core/CHeader.h"
#include "../NiuTensor/source/tensor/function/FHeader.h"
#include <fbgemm/QuantUtils.h>
#include <fbgemm/Fbgemm.h>
//#include <fbgemm/Utils.h>
#include <iostream>
#include <climits>

using namespace nts;
using namespace std;
using namespace fbgemm;

// Memory blocking factors (parameters) for packing into AVX2 int8
static const fbgemm::BlockingFactors Packed8Avx2BlockingFactors = {
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::MR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NR_MIN,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::ROW_INTERLEAVE,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::MCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::KCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NCB
};

// Memory blocking factors (parameters) for packing into AVX512 int8
static const fbgemm::BlockingFactors Packed8Avx512BlockingFactors = {
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NR_MIN,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::ROW_INTERLEAVE,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::KCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NCB
};

enum Type {packed8avx2, packed8avx512};
// This function returns the correct blocking factors structure for given packing type.
inline const fbgemm::BlockingFactors* getBlockingFactors(Type packType) {
  if(packType == Type::packed8avx2) {
    return &Packed8Avx2BlockingFactors;
  } else if(packType == Type::packed8avx512) {
    return &Packed8Avx512BlockingFactors;
  } else {
    printf("Only avx2 and avx512 instruction sets are supported for int8. {}");
  }
}
void printTensor(XTensor & x,int dim)
{
    for(int i=0;i<20;++i)
    {
	    cout<<'=';
    }
    cout<<endl;
    for(int i=0;i<dim;++i)
    {
	    for(int j=0;j<dim;++j)
	    {
		    cout<<x.Get2D(i,j)<<" ";
	    }
	    cout<<endl;
    }
}
int main()
{
	XTensor a,b,bquantized,c;
        int dim=5;
	InitTensor2D(&a, dim, dim, nts::X_FLOAT);
	InitTensor2D(&b, dim, dim, X_FLOAT);
	InitTensor2D(&c, dim, dim,nts::X_INT);
	InitTensor2D(&bquantized, dim, dim, nts::X_INT8);
	a.SetDataRand(2,7);
	b.SetDataRand(2,7);
	TensorQuantizationParams qparams;
        qparams.scale = 1.0;
	qparams.zero_point = 0.0;
	qparams.precision = CHAR_BIT * sizeof(int8_t);
	Quantize<int8_t>((float*)b.data, (int8_t*)bquantized.data,b.unitNum, qparams);

	const BlockingFactors* params = getBlockingFactors(packed8avx2);
	PackBMatrix<int8_t, int32_t> packedBN(
	    matrix_op_t::NoTranspose,
	    dim, dim, (int8_t*)bquantized.data, dim, nullptr, 1, nullptr
			);
	//packedBN.printPackedMatrix("fuck");
	PackAWithQuantRowOffset<uint8_t, int32_t> packAN(
			matrix_op_t::NoTranspose,
			(int32_t)dim,
			(int32_t)dim,
			(const float*)a.data,
			(int32_t)dim,
			nullptr,
			1.0f,
			0,
			1,
			nullptr,
			nullptr
			);
	DoNothing<int32_t, int32_t> doNothingObj{};
	memCopy<> outputProcObj(doNothingObj);
	
	fbgemmPacked(packAN, packedBN, (int32_t*)c.data, (int32_t*)c.data, (int32_t) dim,outputProcObj, 0, 1);
	printTensor(a,dim);
	printTensor(b,dim);
	for(int i=0;i<20;++i)
	{
		cout<<'=';
	}
	cout<<endl;
	for(int i=0;i<dim;++i)
	{
		for(int j=0;j<dim;++j)
		{
			cout<<c.Get2DInt(i,j)<<" ";
		}
		cout<<endl;
	}
}
