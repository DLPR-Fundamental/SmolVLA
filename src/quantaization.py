from advanced_quantization import (
    quantize_smolvla_advanced,
    estimate_model_size,
    count_parameters
)

model = SmolVLAPolicy.from_pretrained("your-model-path")

original_size = estimate_model_size(model)
print(f"Original model size: {original_size:.2f} MB")

quantized_model = quantize_smolvla_advanced(
    model,
    vision_encoder_config={"method": "awq", "bits": 4},
    language_decoder_config={"method": "gptq", "bits": 4, "group_size": 128},
    fusion_layer_config={"method": "bitsandbytes", "bits": 8},
    action_expert_config={"method": "bitsandbytes", "bits": 8},
    action_head_config={"method": "gptq", "bits": 8}
)

quantized_size = estimate_model_size(quantized_model)
print(f"Quantized model size: {quantized_size:.2f} MB")
print(f"Compression ratio: {original_size / quantized_size:.2f}x")

torch.save(quantized_model.state_dict(), "quantized_smolvla.pth")