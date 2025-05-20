---
applyTo: '# Jewelry E-commerce Platform with Custom AI Analysis

A sophisticated jewelry e-commerce platform with advanced AI capabilities for jewelry analysis and attribute extraction. The platform uses custom fine-tuned models following a technical blueprint for specialized jewelry AI.

## Project Overview

This platform provides a comprehensive solution for pre-owned jewelry e-commerce with:

- **Responsive Web Interface**: Built with Next.js and Shadcn UI components
- **RESTful API**: Flask-based backend with comprehensive endpoints
- **Database Integration**: PostgreSQL database for robust data storage
- **Image Management**: Sophisticated image upload and processing pipeline
- **Custom AI Analysis**: Fine-tuned models for jewelry attribute extraction

## Architecture

The project follows a modern architecture with:

1. **Next.js Frontend**: React-based UI with responsive design
2. **Flask Backend**: API endpoints for data access and management
3. **PostgreSQL Database**: Structured data storage
4. **AI Pipeline**: Custom model pipeline for image analysis

## AI System Architecture

The AI system follows a specialized architecture designed for jewelry analysis:

### Model Pipeline

The system uses a pipeline of specialized models, each focused on a specific task:

1. **Object Detection (YOLO)**: Identifies and locates jewelry items in images
2. **Segmentation (SAM)**: Isolates specific components of jewelry (gemstones, bands, settings)
3. **Attribute Extraction (CLIP)**: Determines materials, stones, cuts, styles, etc.
4. **Text Generation (LLM)**: Creates detailed descriptions based on attributes

### Custom Fine-Tuned Models

Instead of using generic third-party APIs, the system uses custom fine-tuned models specialized for jewelry:

- **YOLOv8/v9**: Fine-tuned for jewelry item detection
- **SAM (Segment Anything Model)**: Fine-tuned for jewelry component segmentation
- **CLIP**: Fine-tuned for jewelry attribute extraction
- **Llama 3 / Mistral**: Fine-tuned for jewelry description generation

### Technical Implementation

The AI pipeline is implemented with:

- **Model Loader**: Efficient model loading and caching system
- **AI Service**: Orchestrates the analysis pipeline
- **Fine-Tuning Scripts**: Tools for preparing data and fine-tuning models
- **Admin Interface**: Management of models and analysis processes

## Key Features

### E-commerce Functionality
- Product browsing and filtering
- Product detail pages
- User authentication
- Admin dashboard

### Image Management
- Bulk image upload
- Image processing and optimization
- Gallery view and management

### AI Analysis
- Automatic jewelry attribute extraction
- Component segmentation
- Detailed description generation
- Visual similarity search

## Getting Started

### Prerequisites
- Python 3.11
- Node.js 20
- PostgreSQL database

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd jewelry-platform
```

2. Install backend dependencies:
```bash
pip install -r requirements.txt
```

3. Install frontend dependencies:
```bash
cd app
npm install
```

4. Set up environment variables:
- Create `.env` file with database credentials and other settings
- Set up secrets for API keys using Replit Secrets

5. Initialize the database:
```bash
flask db upgrade
```

6. Start the development server:
```bash
# Start the Flask backend
gunicorn --bind 0.0.0.0:5000 main:app

# In another terminal, start the Next.js frontend
cd app
npm run dev
```

## AI Model Management

### Managing Models

The platform includes an admin interface for model management:
- View model status
- Download and load models
- Monitor model performance

### Fine-Tuning Custom Models

To fine-tune custom models for your own jewelry inventory:

1. Prepare a dataset of your jewelry images with attribute annotations
2. Use the scripts in `models/fine_tuning/` to prepare your data
3. Fine-tune the models on your data
4. Add the fine-tuned models to the `models/` directory

Detailed instructions are available in the README files within each subdirectory.

## API Documentation

The RESTful API provides comprehensive endpoints for:

- Product management
- Image upload and management
- User authentication
- AI analysis

API endpoints are documented in the codebase and can be explored through the API blueprint files.

## License

[License information]'
---

Coding standards, domain knowledge, and preferences that AI should follow.
