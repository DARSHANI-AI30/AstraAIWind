#!/usr/bin/env python3
"""
Startup script for the Weather Prediction Backend
Run this script to start the FastAPI server on port 8000
"""

import uvicorn
import os
import sys

def main():
    """Main function to start the backend server"""
    print("Starting AstraWind AI Backend...")
    print("Weather Prediction API Server")
    print("=" * 50)
    
    # Check if main.py exists
    if not os.path.exists("main.py"):
        print("Error: main.py not found. Please run this script from the backend directory.")
        sys.exit(1)
    
    # Check if dataset exists
    dataset_path = os.path.join("..", "chennai_43279.csv")
    if not os.path.exists(dataset_path):
        print(f"Warning: Dataset not found at {dataset_path}")
        print("The backend will start but some features may not work.")
    
    print(f"Server will be available at: http://localhost:8000")
    print("API Documentation: http://localhost:8000/docs")
    print("=" * 50)
    
    try:
        # Start the server
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info"
        )
    except KeyboardInterrupt:
        print("\nServer stopped by user")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
