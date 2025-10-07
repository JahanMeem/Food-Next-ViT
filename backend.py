import argparse
import torch
from PIL import Image
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models import create_model
import nextvit
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
import uvicorn
from pyngrok import ngrok
import io
import os
from typing import List, Dict
import nest_asyncio

# Apply nest_asyncio to allow nested event loops in Colab
nest_asyncio.apply()

app = FastAPI(title="Food Classification API", version="1.0.0")

# Add CORS middleware to allow requests from React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model
model = None
transform = None
classes = None
device = None

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

def load_model(checkpoint_path, model_name='nextvit_small', num_classes=49, device_name='cuda'):
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
    
    model.to(device_name)
    model.eval()
    
    return model

def predict_image_from_bytes(image_bytes, top_k=5):
    """Predict the class of an image from bytes"""
    global model, transform, classes, device
    
    try:
        # Load and preprocess image
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
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
                'rank': i + 1,
                'class': class_name,
                'probability': float(prob),
                'confidence': float(prob * 100)
            })
        
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.get("/")
async def root():
    """Root endpoint with a simple HTML page"""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Food Classification API</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 50px auto;
                padding: 20px;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
            }
            .container {
                background: rgba(255, 255, 255, 0.95);
                padding: 30px;
                border-radius: 15px;
                color: #333;
                box-shadow: 0 10px 40px rgba(0,0,0,0.3);
            }
            h1 { color: #667eea; margin-top: 0; }
            .status { color: #22c55e; font-weight: bold; }
            .endpoint { 
                background: #f3f4f6; 
                padding: 15px; 
                margin: 10px 0; 
                border-radius: 8px;
                border-left: 4px solid #667eea;
            }
            .method { 
                display: inline-block;
                background: #667eea; 
                color: white; 
                padding: 5px 10px; 
                border-radius: 5px;
                font-size: 12px;
                font-weight: bold;
            }
            a { color: #667eea; text-decoration: none; }
            a:hover { text-decoration: underline; }
            .warning {
                background: #fef3c7;
                border: 2px solid #f59e0b;
                padding: 15px;
                border-radius: 8px;
                margin: 20px 0;
                color: #92400e;
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🍽️ Food Classification API</h1>
            <p class="status">✅ API is running successfully!</p>
            
            <div class="warning">
                <strong>⚠️ Important for Frontend Connection:</strong><br>
                When using the React frontend, make sure to:<br>
                1. Add <code>/api</code> prefix to your URL (e.g., https://your-url.ngrok.io/api)<br>
                2. Or use the endpoints below directly without /api prefix<br>
                3. Visit this page in browser first to bypass ngrok warning
            </div>
            
            <h2>📡 Available Endpoints:</h2>
            
            <div class="endpoint">
                <span class="method">GET</span>
                <strong>/health</strong><br>
                Check API health and model status<br>
                <a href="/health" target="_blank">Try it →</a>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span>
                <strong>/classes</strong><br>
                Get list of all 49 food classes<br>
                <a href="/classes" target="_blank">Try it →</a>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span>
                <strong>/predict</strong><br>
                Predict food class from uploaded image<br>
                Parameters: file (image), top_k (optional, default: 5)
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span>
                <strong>/docs</strong><br>
                Interactive API documentation (Swagger UI)<br>
                <a href="/docs" target="_blank">Open Docs →</a>
            </div>
            
            <h2>🚀 Quick Test:</h2>
            <p>Test the API with cURL:</p>
            <pre style="background: #1f2937; color: #10b981; padding: 15px; border-radius: 8px; overflow-x: auto;">curl -X POST YOUR_URL/predict \\
  -F "file=@/path/to/image.jpg"</pre>
            
            <p style="margin-top: 30px; text-align: center; color: #6b7280;">
                Powered by NextViT Model • Built with FastAPI
            </p>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/api")
async def api_root():
    """API root endpoint"""
    return {
        "message": "Food Classification API is running!",
        "status": "healthy",
        "model": "NextViT Small",
        "num_classes": len(classes) if classes else 0,
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device),
        "num_classes": len(classes) if classes else 0
    }

@app.get("/api/health")
async def api_health_check():
    """Health check endpoint with /api prefix"""
    return await health_check()

@app.get("/classes")
async def get_classes():
    """Get list of all food classes"""
    global classes
    return {
        "classes": classes,
        "total": len(classes)
    }

@app.get("/api/classes")
async def api_get_classes():
    """Get list of all food classes with /api prefix"""
    return await get_classes()

@app.post("/predict")
async def predict(file: UploadFile = File(...), top_k: int = 5):
    """
    Predict food class from uploaded image
    
    Parameters:
    - file: Image file (jpg, jpeg, png)
    - top_k: Number of top predictions to return (default: 5)
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Get predictions
        predictions = predict_image_from_bytes(image_bytes, top_k)
        
        return {
            "success": True,
            "filename": file.filename,
            "predictions": predictions,
            "top_prediction": predictions[0] if predictions else None
        }
    
    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict")
async def api_predict(file: UploadFile = File(...), top_k: int = 5):
    """Predict food class from uploaded image with /api prefix"""
    return await predict(file, top_k)

def initialize_model(checkpoint_path, model_name='nextvit_small', input_size=224, device_name='cuda'):
    """Initialize the model and related components"""
    global model, transform, classes, device
    
    # Check if CUDA is available
    if device_name == 'cuda' and not torch.cuda.is_available():
        print("⚠️  CUDA not available, using CPU")
        device_name = 'cpu'
    
    device = device_name
    
    # Get classes
    classes = get_food_classes()
    num_classes = len(classes)
    
    print(f"📦 Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, model_name, num_classes, device)
    print(f"✅ Model loaded successfully on {device}")
    
    print(f"🔧 Creating transform with input size {input_size}...")
    transform = get_transform(input_size)
    print(f"✅ Transform created successfully")
    
    print(f"📊 Number of classes: {num_classes}")

def start_server(checkpoint_path, model_name='nextvit_small', input_size=224, 
                device_name='cuda', port=8000, ngrok_token=None):
    """Start FastAPI server with ngrok tunnel"""
    
    # Initialize model
    initialize_model(checkpoint_path, model_name, input_size, device_name)
    
    print("\n" + "="*70)
    print("🚀 STARTING FOOD CLASSIFICATION API SERVER")
    print("="*70)
    
    # Set ngrok auth token if provided
    if ngrok_token:
        ngrok.set_auth_token(ngrok_token)
        print(f"✅ ngrok auth token set")
    
    # Start ngrok tunnel
    print(f"\n🌐 Starting ngrok tunnel on port {port}...")
    public_url = ngrok.connect(port)
    print(f"✅ ngrok tunnel established!")
    print(f"\n{'='*70}")
    print(f"🌍 PUBLIC URL: {public_url}")
    print(f"{'='*70}")
    
    print(f"\n⚠️  IMPORTANT SETUP STEPS:")
    print(f"{'='*70}")
    print(f"1. First, visit this URL in your browser:")
    print(f"   {public_url}")
    print(f"   (Click 'Visit Site' on ngrok warning page)")
    print(f"\n2. Then use this URL in your React frontend:")
    print(f"   {public_url}")
    print(f"\n3. Alternative: You can also use with /api prefix:")
    print(f"   {public_url}/api")
    print(f"{'='*70}")
    
    print(f"\n📡 API Endpoints:")
    print(f"   - Home Page:    {public_url}/")
    print(f"   - Health Check: {public_url}/health")
    print(f"   - Get Classes:  {public_url}/classes")
    print(f"   - Predict:      {public_url}/predict (POST)")
    print(f"   - API Docs:     {public_url}/docs")
    print(f"\n{'='*70}")
    print(f"💡 Copy the public URL and paste it in React frontend!")
    print(f"{'='*70}\n")
    
    # Start uvicorn server
    print(f"🔥 Starting Uvicorn server on port {port}...\n")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")

def main():
    parser = argparse.ArgumentParser('Food Classification API with FastAPI and ngrok')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to the model checkpoint')
    parser.add_argument('--model', type=str, default='nextvit_small',
                        help='Model name')
    parser.add_argument('--input-size', type=int, default=224,
                        help='Input image size')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--port', type=int, default=8000,
                        help='Port to run the server on')
    parser.add_argument('--ngrok-token', type=str, default=None,
                        help='ngrok auth token (optional, for custom domains)')
    
    args = parser.parse_args()
    
    start_server(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        input_size=args.input_size,
        device_name=args.device,
        port=args.port,
        ngrok_token=args.ngrok_token
    )

if __name__ == '__main__':
    main()