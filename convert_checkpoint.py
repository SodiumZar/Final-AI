"""
Script to convert old checkpoint to PyTorch 2.6+ compatible format
Run this once to fix your checkpoint file
"""
import torch
from model import build_unet

# Register safe globals for PyTorch version objects
torch.serialization.add_safe_globals([torch.torch_version.TorchVersion])

# Load old checkpoint with weights_only=False
print("Loading old checkpoint...")
old_checkpoint = torch.load('files/checkpoint.pth', map_location='cpu', weights_only=False)

# Extract only the state_dict (pure tensor weights)
if isinstance(old_checkpoint, dict):
    if 'state_dict' in old_checkpoint:
        state_dict = old_checkpoint['state_dict']
    elif 'model_state_dict' in old_checkpoint:
        state_dict = old_checkpoint['model_state_dict']
    else:
        # Assume the whole dict is the state_dict
        state_dict = old_checkpoint
else:
    # If the checkpoint is the model object itself
    state_dict = old_checkpoint.state_dict()

print(f"Found {len(state_dict)} weight tensors")

# Create new clean checkpoint with ONLY tensors (no metadata)
print("Creating new compatible checkpoint...")
new_checkpoint = {
    'state_dict': state_dict
}

# Save new checkpoint
new_path = 'files/checkpoint_compatible.pth'
torch.save(new_checkpoint, new_path)
print(f"✓ New checkpoint saved to: {new_path}")

# Verify it can be loaded with weights_only=True
print("\nVerifying new checkpoint...")
try:
    test_load = torch.load(new_path, weights_only=True)
    print("✓ New checkpoint is compatible with PyTorch 2.6+")
    
    # Test loading into model
    model = build_unet()
    model.load_state_dict(test_load['state_dict'])
    print("✓ Model loaded successfully")
    
    print("\n" + "="*60)
    print("SUCCESS! To use the new checkpoint, update predict.py:")
    print('Change: checkpoint_path="files/checkpoint_compatible.pth"')
    print("="*60)
    
except Exception as e:
    print(f"⚠ Warning during verification: {e}")
    print("\nBut the checkpoint should still work with weights_only=False")
    print("Testing fallback load...")
    
    try:
        test_load = torch.load(new_path, weights_only=False)
        model = build_unet()
        model.load_state_dict(test_load['state_dict'])
        print("✓ Checkpoint works with weights_only=False")
        print("\nThe updated predict.py will handle this automatically.")
    except Exception as e2:
        print(f"✗ Error: {e2}")
