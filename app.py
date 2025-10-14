# Root shim: redirect to the modular app
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.dirname(__file__))

# Import and run the main function from the modular app
from app.app import main

if __name__ == "__main__":
    main()
