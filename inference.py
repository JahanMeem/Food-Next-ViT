import argparse
import torch
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
import nextvit

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
    # Create model
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
        print(f"Warning: Error loading checkpoint with weights_only=False: {e}")
        print("Trying alternative loading method...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model

def predict_image(image_path, model, transform, classes, device='cuda', top_k=5):
    """Predict the class of a single image"""
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top-k predictions
    top_prob, top_indices = torch.topk(probabilities, top_k)
    
    results = []
    for i in range(top_k):
        class_idx = top_indices[i].item()
        prob = top_prob[i].item()
        class_name = classes[class_idx]
        results.append({
            'class': class_name,
            'probability': prob,
            'confidence': prob * 100
        })
    
    return results

def main():
    parser = argparse.ArgumentParser('Food Image Classification Inference')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the image file')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--model', type=str, default='nextvit_small',
                        help='Model name')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--top-k', type=int, default=5,
                        help='Show top-k predictions')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        args.device = 'cpu'
    
    # Get classes
    classes = get_food_classes()
    num_classes = len(classes)
    
    print(f"Loading model from {args.checkpoint}...")
    model = load_model(args.checkpoint, args.model, num_classes, args.device)
    
    print(f"Creating transform with input size {args.input_size}...")
    transform = get_transform(args.input_size)
    
    print(f"\nPredicting image: {args.image}")
    results = predict_image(args.image, model, transform, classes, args.device, args.top_k)
    
    print("\n" + "="*60)
    print("PREDICTION RESULTS")
    print("="*60)
    
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['class']:<30} {result['confidence']:>6.2f}%")
    
    print("="*60)
    print(f"\nTop prediction: {results[0]['class']} ({results[0]['confidence']:.2f}% confidence)")
    print("="*60)

if __name__ == '__main__':
    main()