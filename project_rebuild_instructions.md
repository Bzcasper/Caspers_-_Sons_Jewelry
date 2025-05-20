# Project Rebuild Instructions

Please provide step-by-step instructions and code for creating a Mobile App project titled "AI-Powered Jewelry Analysis API & Mobile App" using Python (Flask/FastAPI), PostgreSQL, Modal Labs for the backend and React Native with TypeScript for the mobile app that can be set up in Replit. 

## Backend API (Python)

The backend should include:

1. **Setup Flask/FastAPI with PostgreSQL**
   - Configure RESTful API endpoints
   - Set up database models for jewelry items, users, and search history
   - Implement authentication with JWT

2. **AI Image Processing Pipeline**
   - Integration with Modal Labs for GPU-accelerated processing
   - Four sequential model pipeline:
     - Object detection (YOLOv8x)
     - Segmentation (SAM)
     - Attribute extraction (CLIP-ViT)
     - Description generation (GPT-4o)
   - Processing queue for handling batch uploads

3. **Vector Search Implementation**
   - Store and index embeddings for jewelry images
   - Implement similarity search API endpoints
   - Support for multi-attribute filtering

## Mobile App (React Native with TypeScript)

The mobile app should include:

1. **User Interface**
   - Home screen with featured jewelry items
   - Search functionality with filters
   - Product detail pages with AI-generated descriptions
   - Camera integration for capturing jewelry images

2. **State Management**
   - Redux store configuration
   - API integration services
   - Authentication flow

3. **Visual Search Feature**
   - Image capture/upload component
   - Results display with similarity scores
   - "More like this" recommendations

## Integration Requirements

- Secure API communication
- Real-time processing status updates
- Offline capabilities for basic browsing
- Push notifications for processing completion

## Environment Setup

- Required environment variables for API keys and database connection
- Setup instructions for Modal Labs integration
- Development and production configurations