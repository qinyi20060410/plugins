#include "mish.h"
#include <cassert>
#include <cmath>
#include <iostream>
#include <stdio.h>

namespace nvinfer1
{

namespace plugin
{

mishPlugin::mishPlugin() {}

mishPlugin::~mishPlugin() {}

// create the plugin at runtime from a byte stream
mishPlugin::mishPlugin(const void* data, size_t length)
{
    assert(length == sizeof(input_size_));
    input_size_ = *reinterpret_cast<const int*>(data);
}

void mishPlugin::serialize(void* buffer) const
{
    *reinterpret_cast<int*>(buffer) = input_size_;
}

size_t mishPlugin::getSerializationSize() const
{
    return sizeof(input_size_);
}

int mishPlugin::initialize()
{
    return 0;
}

Dims mishPlugin::getOutputDimensions(int index, const Dims* inputs, int nbInputDims)
{
    assert(nbInputDims == 1);
    assert(index == 0);
    input_size_ = inputs[0].d[0] * inputs[0].d[1] * inputs[0].d[2];
    // Output dimensions
    return Dims3(inputs[0].d[0], inputs[0].d[1], inputs[0].d[2]);
}

// Set plugin namespace
void mishPlugin::setPluginNamespace(const char* pluginNamespace)
{
    mPluginNamespace = pluginNamespace;
}

const char* mishPlugin::getPluginNamespace() const
{
    return mPluginNamespace;
}

// Return the DataType of the plugin output at the requested index
DataType mishPlugin::getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const
{
    return DataType::kFLOAT;
}

// Return true if output tensor is broadcast across a batch.
bool mishPlugin::isOutputBroadcastAcrossBatch(int outputIndex, const bool* inputIsBroadcasted, int nbInputs) const
{
    return false;
}

// Return true if plugin can use input that is broadcast across batch without replication.
bool mishPlugin::canBroadcastInputAcrossBatch(int inputIndex) const
{
    return false;
}

void mishPlugin::configurePlugin(const PluginTensorDesc* in, int nbInput, const PluginTensorDesc* out, int nbOutput) {}

// Attach the plugin object to an execution context and grant the plugin the access to some context resource.
void mishPlugin::attachToContext(cudnnContext* cudnnContext, cublasContext* cublasContext, IGpuAllocator* gpuAllocator)
{
}

// Detach the plugin object from its execution context.
void mishPlugin::detachFromContext() {}

const char* mishPlugin::getPluginType() const
{
    return "Mish_TRT";
}

const char* mishPlugin::getPluginVersion() const
{
    return "1";
}

void mishPlugin::destroy()
{
    delete this;
}

// Clone the plugin
IPluginV2IOExt* mishPlugin::clone() const
{
    mishPlugin* p = new mishPlugin();
    p->input_size_ = input_size_;
    p->setPluginNamespace(mPluginNamespace);
    return p;
}

__device__ float tanh_activate_kernel(float x)
{
    return (2 / (1 + expf(-2 * x)) - 1);
}

__device__ float softplus_kernel(float x, float threshold = 20)
{
    if (x > threshold)
        return x; // too large
    else if (x < -threshold)
        return expf(x); // too small
    return logf(expf(x) + 1);
}

__global__ void mish_kernel(const float* input, float* output, int num_elem)
{

    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= num_elem)
        return;

    // float t = exp(input[idx]);
    // if (input[idx] > 20.0) {
    //    t *= t;
    //    output[idx] = (t - 1.0) / (t + 1.0);
    //} else {
    //    float tt = t * t;
    //    output[idx] = (tt + 2.0 * t) / (tt + 2.0 * t + 2.0);
    //}
    // output[idx] *= input[idx];
    output[idx] = input[idx] * tanh_activate_kernel(softplus_kernel(input[idx]));
}

void mishPlugin::forwardGpu(const float* const* inputs, float* output, cudaStream_t stream, int batchSize)
{
    int block_size = thread_count_;
    int grid_size = (input_size_ * batchSize + block_size - 1) / block_size;
    mish_kernel<<<grid_size, block_size>>>(inputs[0], output, input_size_ * batchSize);
}

int mishPlugin::enqueue(int batchSize, const void* const* inputs, void** outputs, void* workspace, cudaStream_t stream)
{
    // assert(batchSize == 1);
    // GPU
    // CUDA_CHECK(cudaStreamSynchronize(stream));
    forwardGpu((const float* const*) inputs, (float*) outputs[0], stream, batchSize);
    return 0;
}

PluginFieldCollection mishPluginCreator::mFC{};
std::vector<PluginField> mishPluginCreator::mPluginAttributes;

mishPluginCreator::mishPluginCreator()
{
    mPluginAttributes.clear();

    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

const char* mishPluginCreator::getPluginName() const
{
    return "Mish_TRT";
}

const char* mishPluginCreator::getPluginVersion() const
{
    return "1";
}

const PluginFieldCollection* mishPluginCreator::getFieldNames()
{
    return &mFC;
}

IPluginV2IOExt* mishPluginCreator::createPlugin(const char* name, const PluginFieldCollection* fc)
{
    mishPlugin* obj = new mishPlugin();
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}

IPluginV2IOExt* mishPluginCreator::deserializePlugin(const char* name, const void* serialData, size_t serialLength)
{
    // This object will be deleted when the network is destroyed, which will
    // call mishPlugin::destroy()
    mishPlugin* obj = new mishPlugin(serialData, serialLength);
    obj->setPluginNamespace(mNamespace.c_str());
    return obj;
}
} // namespace plugin
} // namespace nvinfer1
