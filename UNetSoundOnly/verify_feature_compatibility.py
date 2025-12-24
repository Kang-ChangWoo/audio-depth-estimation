"""
Verify Feature Compatibility Between RGB and Binaural Models

This script verifies that the RGB depth model and binaural attention model
produce compatible feature dimensions at all levels, which is essential
for knowledge distillation.

Usage:
    python verify_feature_compatibility.py
"""

import torch
from models.rgb_depth_model import create_rgb_depth_model
from models.binaural_attention_model import create_binaural_attention_model


def print_header(title):
    """Print a formatted header"""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def print_feature_info(features, model_name):
    """Print feature dimensions"""
    print(f"\n{model_name} Features:")
    print("-" * 80)
    for key, feat in features.items():
        if isinstance(feat, torch.Tensor):
            print(f"  {key:8s}: {str(list(feat.shape)):30s} "
                  f"({feat.numel():,} elements)")


def compare_features(rgb_features, audio_features):
    """Compare feature dimensions between RGB and audio models"""
    print_header("Feature Compatibility Check")
    
    compatible = True
    issues = []
    
    # Check encoder features
    for level in ['x1', 'x2', 'x3', 'x4', 'x5']:
        if level in rgb_features and level in audio_features:
            rgb_shape = rgb_features[level].shape
            audio_shape = audio_features[level].shape
            
            match = rgb_shape == audio_shape
            status = "✅" if match else "❌"
            
            print(f"\nLevel {level}:")
            print(f"  RGB:   {list(rgb_shape)}")
            print(f"  Audio: {list(audio_shape)}")
            print(f"  Match: {status}")
            
            if not match:
                compatible = False
                issues.append(f"Level {level}: RGB {rgb_shape} != Audio {audio_shape}")
        else:
            print(f"\n⚠️  Level {level} missing in one model")
            compatible = False
    
    print("\n" + "=" * 80)
    if compatible:
        print("✅ ALL FEATURES COMPATIBLE!")
        print("   Models are ready for knowledge distillation.")
    else:
        print("❌ COMPATIBILITY ISSUES FOUND:")
        for issue in issues:
            print(f"   - {issue}")
    print("=" * 80)
    
    return compatible


def verify_compatibility():
    """Main verification function"""
    
    print_header("Model Feature Compatibility Verification")
    print("\nThis script verifies that RGB and binaural models produce")
    print("compatible feature dimensions for knowledge distillation.\n")
    
    # Configuration
    base_channels = 64
    bilinear = True
    output_size = 256
    max_depth = 30.0
    batch_size = 2
    
    print(f"Configuration:")
    print(f"  Base channels: {base_channels}")
    print(f"  Bilinear: {bilinear}")
    print(f"  Output size: {output_size}x{output_size}")
    print(f"  Batch size: {batch_size}")
    
    # ==========================================
    # Create Models
    # ==========================================
    print_header("Creating Models")
    
    print("\n1. Creating RGB Depth Model...")
    rgb_model = create_rgb_depth_model(
        base_channels=base_channels,
        bilinear=bilinear,
        output_size=output_size,
        max_depth=max_depth
    )
    rgb_model.eval()
    
    print("\n2. Creating Binaural Attention Model...")
    audio_model = create_binaural_attention_model(
        base_channels=base_channels,
        bilinear=bilinear,
        output_size=output_size,
        max_depth=max_depth,
        attention_levels=[2, 3, 4, 5]
    )
    audio_model.eval()
    
    # ==========================================
    # Create Dummy Inputs
    # ==========================================
    print_header("Creating Dummy Inputs")
    
    rgb_input = torch.randn(batch_size, 3, output_size, output_size)
    audio_input = torch.randn(batch_size, 2, output_size, output_size)
    
    print(f"\nRGB input shape:   {list(rgb_input.shape)}")
    print(f"Audio input shape: {list(audio_input.shape)}")
    
    # ==========================================
    # Forward Pass (RGB Model)
    # ==========================================
    print_header("RGB Model Forward Pass")
    
    with torch.no_grad():
        rgb_depth, rgb_features = rgb_model(rgb_input, return_features=True)
    
    print(f"\nRGB depth output: {list(rgb_depth.shape)}")
    print_feature_info(rgb_features, "RGB Model")
    
    # ==========================================
    # Forward Pass (Audio Model)
    # ==========================================
    print_header("Binaural Audio Model Forward Pass")
    
    # Extract features from binaural model
    # Note: The binaural model returns fused features after attention
    with torch.no_grad():
        # Get intermediate features by modifying forward pass
        audio_model.eval()
        
        # Split into left and right channels
        left = audio_input[:, 0:1, :, :]
        right = audio_input[:, 1:2, :, :]
        
        # Encode separately
        left_feats = audio_model.left_encoder(left)
        right_feats = audio_model.right_encoder(right)
        
        # Apply attention and fusion (same as in model.forward)
        audio_features = {}
        for level in [1, 2, 3, 4, 5]:
            left_feat = left_feats[f'x{level}']
            right_feat = right_feats[f'x{level}']
            
            if level in audio_model.attention_levels:
                # Apply attention
                left_feat, right_feat = audio_model.attention_modules[f'attn_{level}'](
                    left_feat, right_feat
                )
            
            # Fuse left and right (this is what we compare with RGB)
            fused = audio_model.fusion_layers[f'fusion_{level}'](
                torch.cat([left_feat, right_feat], dim=1)
            )
            audio_features[f'x{level}'] = fused
        
        # Get final depth output
        audio_depth = audio_model(audio_input)
    
    print(f"\nAudio depth output: {list(audio_depth.shape)}")
    print_feature_info(audio_features, "Binaural Audio Model (After Fusion)")
    
    # ==========================================
    # Compare Features
    # ==========================================
    compatible = compare_features(rgb_features, audio_features)
    
    # ==========================================
    # Additional Checks
    # ==========================================
    print_header("Additional Verification")
    
    # Check depth output shapes
    print("\nDepth Output Comparison:")
    print(f"  RGB depth:   {list(rgb_depth.shape)}")
    print(f"  Audio depth: {list(audio_depth.shape)}")
    depth_match = rgb_depth.shape == audio_depth.shape
    print(f"  Match: {'✅' if depth_match else '❌'}")
    
    # Check parameter counts
    print("\nParameter Counts:")
    rgb_params = rgb_model.get_num_params()
    audio_params = audio_model.get_num_params()
    print(f"  RGB model:   {rgb_params:,} parameters")
    print(f"  Audio model: {audio_params:,} parameters")
    print(f"  Ratio:       {audio_params / rgb_params:.2f}x")
    print(f"  Note:        Audio model is larger due to dual encoders + attention")
    
    # ==========================================
    # Summary
    # ==========================================
    print_header("Summary")
    
    if compatible and depth_match:
        print("\n✅ SUCCESS: Models are fully compatible!")
        print("\nYou can now proceed with knowledge distillation:")
        print("  1. Train RGB model as teacher")
        print("  2. Use RGB features to guide audio model training")
        print("  3. Match features at levels x1, x2, x3, x4, x5")
        print("\nExample distillation loss:")
        print("  loss_kd = 0")
        print("  for level in ['x1', 'x2', 'x3', 'x4', 'x5']:")
        print("      loss_kd += F.mse_loss(audio_feats[level], rgb_feats[level])")
    else:
        print("\n❌ FAILURE: Models have compatibility issues!")
        print("\nPlease fix the following:")
        if not compatible:
            print("  - Feature dimension mismatches")
        if not depth_match:
            print("  - Depth output dimension mismatch")
    
    print("\n" + "=" * 80)
    
    return compatible and depth_match


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("  FEATURE COMPATIBILITY VERIFICATION")
    print("  RGB Depth Model ↔ Binaural Attention Model")
    print("=" * 80)
    
    try:
        success = verify_compatibility()
        exit_code = 0 if success else 1
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        exit_code = 1
    
    print("\n" + "=" * 80)
    if exit_code == 0:
        print("✅ Verification completed successfully!")
    else:
        print("❌ Verification failed!")
    print("=" * 80 + "\n")
    
    exit(exit_code)






