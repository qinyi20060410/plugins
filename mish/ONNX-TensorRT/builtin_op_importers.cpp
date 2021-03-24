
DEFINE_BUILTIN_OP_IMPORTER(Mish) {
  ASSERT(inputs.at(0).is_tensor(), ErrorCode::kUNSUPPORTED_NODE);  // input

  // data
  nvinfer1::ITensor* tensorPtr = &convertToTensor(inputs.at(0), ctx);

  // plugin
  const std::string pluginName = "Mish_TRT";
  const std::string pluginVersion = "1";
  std::vector<nvinfer1::PluginField> f;

  // Create plugin from registry
  nvinfer1::IPluginV2* plugin = createPlugin(
      node.name(), importPluginCreator(pluginName, pluginVersion), f);

  auto* layer = ctx->network()->addPluginV2(&tensorPtr, 1, *plugin);
  ctx->registerLayer(layer, node.name());
  RETURN_FIRST_OUTPUT(layer);
}