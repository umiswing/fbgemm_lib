#include "../NiuTensor/source/tensor/XTensor.h"
#include "../NiuTensor/source/tensor/core/CHeader.h"
#include "../NiuTensor/source/tensor/function/FHeader.h"
#include <fbgemm/QuantUtils.h>
#include <fbgemm/Fbgemm.h>
#include <iostream>

using namespace nts;
using namespace fbgemm;
using namespace std;

enum Type {packed8avx2, packed8avx512};

// Memory blocking factors (parameters) for packing into AVX2 int8
static const BlockingFactors Packed8Avx2BlockingFactors = {
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::MR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NR_MIN,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::ROW_INTERLEAVE,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::MCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::KCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx2>::NCB
};

// Memory blocking factors (parameters) for packing into AVX512 int8
static const BlockingFactors Packed8Avx512BlockingFactors = {
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NR,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NR_MIN,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::ROW_INTERLEAVE,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::MCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::KCB,
    PackingTraits<int8_t, int32_t, inst_set_t::avx512>::NCB
};

// This function returns the correct blocking factors structure for given packing type.
inline const BlockingFactors* getBlockingFactors(Type packType) {
  if(packType == Type::packed8avx2) {
    return &Packed8Avx2BlockingFactors;
  } else if(packType == Type::packed8avx512) {
    return &Packed8Avx512BlockingFactors;
  } else {
    //ABORT("Only avx2 and avx512 instruction sets are supported for int8. {}", packType);
  }
}

void fbgemmPacked8PackInfo(const vector<int>& shape,
                           const Type packType,
                           const bool transpose,
                           int& nrow,
                           int& ncol,
                           uint64_t& packsize) {
    // Should be 2D - weight matrix
    //ABORT_IF(shape.size() != 2,
            //"Weight Matrix should be 2D");
    nrow = transpose ? shape[1] : shape[0];
    ncol = transpose ? shape[0] : shape[1];

    const BlockingFactors* params =getBlockingFactors(packType);

    packsize = fbgemm::PackMatrix<fbgemm::PackBMatrix<int8_t>, int8_t>::packedBufferSize(
        transpose ? shape[1] : shape[0],
        transpose ? shape[0] : shape[1], params);
    // add extra space for storing some other variables specific to B matrix
    // quantization sacles: 1 per column and float
    // quantization offset: 1 per column and int32
    // column offsets: 1 per column and int32
    packsize += ncol * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t));
}

// This function computes the offset values for each column which are used for compensating the remainders of quantized values
// More detailed math is avilable in the FBGEMM's blog - https://engineering.fb.com/ml-applications/fbgemm/
inline void col_offsets_with_zero_pt_s8acc32(
    bool transpose,
    int K,
    int N,
    const int8_t* Bint8,
    const int32_t* B_zero_point,
    int32_t* col_offsets,
    int ncols_per_quant_group) {
  for (int n = 0; n < N; ++n) {
    int32_t sum = 0;
    for (int k = 0; k < K; ++k) {
      sum += transpose ? Bint8[k + n * K] : Bint8[k * N + n];
    }
    col_offsets[n] = sum - B_zero_point[n / ncols_per_quant_group] * K;
  }
}
void fbgemmPacked8Pack(
		//XTensor out,
		int8_t* packedbuf,
		const float* inData,
		const Type packType,
		const bool transpose,
		const int nrow,
		const int ncol,
		const uint64_t packsize) {
	int k = nrow;
	int n = ncol;
	int len = k * n;

	// 1. collect stats for each column
	float* bqScale = new float[n];
	int32_t* bqZeropoint = new int32_t[n];

	const float* data = inData;
	float val = 0;

	if (transpose) {
		for (int jj = 0; jj < n; jj++) {
			float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
			double mean = 0, sqrsum = 0;
			for (int ii = 0; ii < k; ii++) {
				val = data[jj * k + ii];
				mean += val;
				sqrsum += val * val;
			}
			mean /= k;
			sqrsum /= k;
			sqrsum -= mean * mean;
			sqrsum = sqrt(sqrsum);

			min = (float)(mean - 7.0f*sqrsum);
			max = (float)(mean + 7.0f*sqrsum);
			bqScale[jj] = (max - min) / 255;
			bqZeropoint[jj] = (int32_t)(127 - max / bqScale[jj]);
		}
	} else {
		for (int jj = 0; jj < n; jj++) {
			float min = std::numeric_limits<float>::max(), max = std::numeric_limits<float>::min();
			double mean = 0, sqrsum = 0;
			for (int ii = 0; ii < k; ii++) {
				val = data[jj + ii * n];
				mean += val;
				sqrsum += val * val;
			}
			mean /= k;
			sqrsum /= k;
			sqrsum -= mean * mean;
			sqrsum = sqrt(sqrsum);

			min = (float)(mean - 7.0f*sqrsum);
			max = (float)(mean + 7.0f*sqrsum);
			bqScale[jj] = (max - min) / 255;
			bqZeropoint[jj] = (int32_t)(127 - max / bqScale[jj]);
		}
	}
	
  // 2. quantize
  int8_t* quantized = 0;
#ifdef _MSC_VER
  quantized = (int8_t*)_aligned_malloc(len, 256);
#else
  int result = posix_memalign((void**)&quantized, 256, len); result;
  assert(result == 0);
#endif
  for (int jj = 0; jj < n; jj++) {
    TensorQuantizationParams bQuantParam;
    bQuantParam.scale = bqScale[jj];
    bQuantParam.zero_point = bqZeropoint[jj];
    bQuantParam.precision = 8;

    if (transpose)
      fbgemm::Quantize<int8_t>(data + jj * k, quantized + jj * k, k, bQuantParam);
    else {
      for (int ii = 0; ii < k; ii++) {
        quantized[ii*n + jj] = fbgemm::Quantize<int8_t>(data[ii*n + jj], bQuantParam);
      }
    }
  }

  // 3. compute column offsets
  int32_t* col_offsets = new int32_t[n];
  col_offsets_with_zero_pt_s8acc32(transpose, k, n, quantized, bqZeropoint, col_offsets, 1);


  //int8_t* packedbuf = (int8_t*)out.data;
   packedbuf = (int8_t*)malloc((size_t)packsize);
  for(auto i = 0; i < packsize; i++) {
    packedbuf[i] = 0;
  }

  // 4. packing
  const fbgemm::BlockingFactors* params = getBlockingFactors(packType);
  
  PackBMatrix<int8_t> packedBN(
      transpose ? matrix_op_t::Transpose : matrix_op_t::NoTranspose,
      nrow, ncol, quantized, transpose ? nrow : ncol, packedbuf, 1, params);
  packedBN.printPackedMatrix("fuck");

  // copy quantization scale
  memcpy(packedbuf + (packsize - n * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t))), bqScale, n * sizeof(float));
  // copy quantization offset
  memcpy(packedbuf + (packsize - n * (sizeof(int32_t) + sizeof(int32_t))), bqZeropoint, n * sizeof(int32_t));
  // copy column offsets to the memory
  memcpy(packedbuf + (packsize - n * sizeof(int32_t)), col_offsets, n * sizeof(int32_t));

#ifdef _MSC_VER
  _aligned_free(quantized);
#else
  free(quantized);
#endif
  delete[] col_offsets;
  delete[] bqScale;
  delete[] bqZeropoint;
}

void printTensor(XTensor & x,int row, int col)
{
    for(int i=0;i<20;++i)
    {
	    cout<<'=';
    }
    cout<<endl;
    for(int i=0;i<row;++i)
    {
	    for(int j=0;j<col;++j)
	    {
		    cout<<x.Get2D(i,j)<<" ";
	    }
	    cout<<endl;
    }
}

void printTensorInt8(XTensor & x,int row, int col)
{
    for(int i=0;i<20;++i)
    {
	    cout<<'=';
    }
    cout<<endl;
    for(int i=0;i<row;++i)
    {
	    for(int j=0;j<col;++j)
	    {
		    cout<<static_cast<int16_t>(x.Get2DInt8(i,j))<<" ";
	    }
	    cout<<endl;
    }
}
int main()
{
	XTensor b;
	int8_t* bPacked=nullptr;
	int dim=5;
	vector<int> shape;
	shape.push_back(dim);
	shape.push_back(dim);
	int nrow,ncol;
	uint64_t packsize;
	fbgemmPacked8PackInfo(
			shape,
			packed8avx2,
			false,
			nrow,
			ncol,
			packsize
			);
	InitTensor2D(&b, dim, dim, nts::X_FLOAT);
	b.SetDataRand(2,7);
	printTensor(b,dim,dim);
	fbgemmPacked8Pack(
			bPacked,
			(float*)b.data,
			Type::packed8avx2,
			false,
			nrow,
			ncol,
			packsize
			);
	//printTensorInt8(bPacked,nrow,ncol);
	
	for(auto i=0;i<(packsize - ncol * (sizeof(float) + sizeof(int32_t) + sizeof(int32_t)));++i)
	{
        	cout<<static_cast<int16_t>(bPacked[i])<<"  ";
	}
	cout<<nrow<<' '<<ncol<<endl;
}
