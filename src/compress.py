import torch
import torch.nn.utils.prune as prune
import os
import argparse
from models.snunet import SNUNet

def prune_model(model, amount=0.3):
    """
    Prunes the model globally.
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            parameters_to_prune.append((module, 'weight'))
    
    # Global pruning of L1 unstructured
    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=amount,
    )
    
    # Remove reparameterization to make pruning permanent
    for module, name in parameters_to_prune:
        prune.remove(module, name)
        
    return model

def quantize_model(model):
    """
    Dynamic quantization for CPU inference.
    """
    # Dynamic quantization only works for Linear and RNN/LSTM on CPU in standard PyTorch 
    # (Conv2d dynamic quantization is limited or requires QAT).
    # For Conv2d, we usually use Static Quantization. 
    # However, let's try dynamic for Linear first or full static int8.
    
    # Simplified approach: Dynamic (Int8)
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.LSTM, torch.nn.Linear}, dtype=torch.qint8
    )
    return quantized_model

def main(args):
    device = torch.device('cpu') # Quantization often targets CPU
    
    # Load model
    model = SNUNet(3, 1, use_attention=True)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Original model loaded from {args.model_path}")
    
    # 1. Pruning
    print(f"Pruning model with amount={args.prune_amount}...")
    model = prune_model(model, amount=args.prune_amount)
    
    # Save pruned
    pruned_path = args.model_path.replace('.pth', '_pruned.pth')
    torch.save(model.state_dict(), pruned_path)
    print(f"Pruned model saved to {pruned_path}")
    
    # 2. Quantization (Simple Dynamic)
    # Note: For SNUNet (mostly Conv2d), dynamic quantization of Linear layers might not yield much status.
    # But we implemented it for demonstrating the pipeline.
    print("Quantizing model...")
    quantized_model = quantize_model(model)
    
    quantized_path = args.model_path.replace('.pth', '_quantized.pth')
    torch.save(quantized_model.state_dict(), quantized_path)
    print(f"Quantized model saved to {quantized_path}")
    
    # Size comparison
    size_orig = os.path.getsize(args.model_path) / 1e6
    size_pruned = os.path.getsize(pruned_path) / 1e6
    size_quant = os.path.getsize(quantized_path) / 1e6
    
    print(f"Original Size: {size_orig:.2f} MB")
    print(f"Pruned Size: {size_pruned:.2f} MB")
    print(f"Quantized Size: {size_quant:.2f} MB")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True, help='Path to .pth model file')
    parser.add_argument('--prune_amount', type=float, default=0.3, help='Fraction of weights to prune')
    args = parser.parse_args()
    
    main(args)
