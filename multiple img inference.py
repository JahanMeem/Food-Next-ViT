import argparse
import torch
import os
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
import nextvit
from pathlib import Path
import pandas as pd
from tqdm import tqdm

def get_food_classes():
    """Return the list of food categories in your dataset"""
    return [
        'Biriyani', 'Boiled_egg', 'Buter_Dal', 'Cake', 'Cha', 
        'Chicken_curry', 'Chicken_wings', 'Chocolate_cake', 'Chow_mein', 
        'Crab_Dish_Kakra', 'Fish Bhuna_Mach Bhuna', 'French_fries', 
        'Fried fish_Mach Bhaja', 'Fried_rice', 'Hilsha_Fish_Curry', 
        'Kacchi', 'Khichuri', 'Lentil fritters_Dal Puri', 'Lentil soup_Dal', 
        'Meat Curry_Gosht Bhuna', 'Misti', 'Momos', 'Naan Ruti', 
        'Rosogolla', 'Salad', 'Sandwich', 'Shik_kabab', 'Singgara', 
        'Vegetable fritters _Beguni', 'Vorta', 'bakorkhani', 'cheesecake', 
        'cup_cakes', 'fuchka', 'golap Jam', 'haleem', 'ice_cream', 
        'jilapi', 'kebab_Gosht Kebab', 'morog_polao', 'nehari', 
        'omelette', 'pakora', 'pizza', 'poached_egg', 'porota', 
        'roshmalai', 'steak', 'yogurt'
    ]

def get_transform(input_size=224):
    """Create the same transform used during validation"""
    size = int((256 / 224) * input_size)
    t = transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD)
    ])
    return t

def load_model(checkpoint_path, model_name='nextvit_small', num_classes=49, device='cuda'):
    """Load the trained model from checkpoint"""
    model = create_model(
        model_name,
        num_classes=num_classes,
    )
    
    # Load checkpoint with safe globals for argparse.Namespace
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])
    
    try:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    except Exception as e:
        print(f"Warning: Error loading checkpoint: {e}")
        print("Trying alternative loading method...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

def predict_images_batch(image_paths, model, transform, classes, device='cuda', batch_size=32):
    """Predict classes for multiple images"""
    results = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Processing batches"):
        batch_paths = image_paths[i:i+batch_size]
        batch_images = []
        valid_paths = []
        
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                image_tensor = transform(image)
                batch_images.append(image_tensor)
                valid_paths.append(img_path)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")
                continue
        
        if not batch_images:
            continue
        
        batch_tensor = torch.stack(batch_images).to(device)
        
        with torch.no_grad():
            outputs = model(batch_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            top_prob, top_indices = torch.max(probabilities, dim=1)
        
        for j, img_path in enumerate(valid_paths):
            class_idx = top_indices[j].item()
            prob = top_prob[j].item()
            
            results.append({
                'image_path': img_path,
                'predicted_class': classes[class_idx],
                'confidence': prob * 100
            })
    
    return results

def main():
    parser = argparse.ArgumentParser('Batch Food Image Classification')
    parser.add_argument('--image-dir', type=str,
                        help='Directory containing images')
    parser.add_argument('--image-list', type=str,
                        help='Text file with list of image paths (one per line)')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--model', type=str, default='nextvit_small',
                        help='Model name')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--output-csv', type=str, default='predictions.csv',
                        help='Output CSV file path')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Get image paths
    image_paths = []
    if args.image_dir:
        image_dir = Path(args.image_dir)
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
            image_paths.extend(list(image_dir.glob(ext)))
        image_paths = [str(p) for p in image_paths]
    elif args.image_list:
        with open(args.image_list, 'r') as f:
            image_paths = [line.strip() for line in f if line.strip()]
    else:
        print("Error: Please provide either --image-dir or --image-list")
        return
    
    if not image_paths:
        print("No images found!")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Get classes
    classes = get_food_classes()
    num_classes = len(classes)
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model, num_classes, args.device)
    
    print(f"Creating transform with input size {args.input_size}...")
    transform = get_transform(args.input_size)
    
    print("\nStarting batch prediction...")
    results = predict_images_batch(
        image_paths, model, transform, classes, args.device, args.batch_size
    )
    
    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv(args.output_csv, index=False)
    print(f"\nResults saved to {args.output_csv}")
    
    # Print summary
    print("\n" + "="*60)
    print("PREDICTION SUMMARY")
    print("="*60)
    print(f"Total images processed: {len(results)}")
    print(f"Average confidence: {df['confidence'].mean():.2f}%")
    print("\nClass distribution:")
    print(df['predicted_class'].value_counts())
    print("="*60)

if __name__ == '__main__':
    main()