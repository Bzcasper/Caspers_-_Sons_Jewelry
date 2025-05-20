# AI-Powered Jewelry E-Commerce Platform

A comprehensive jewelry cataloging and e-commerce platform with advanced AI-powered image analysis capabilities. The platform processes jewelry images through a sequence of specialized AI models to automatically extract attributes, generate descriptions, and create product listings without manual data entry.

## Features

### AI Image Analysis Pipeline

The platform processes jewelry images through four sequential specialized AI models:
- **Object Detection**: Uses YOLOv8x to identify and locate jewelry items in images
- **Segmentation**: Employs SAM (Segment Anything Model) to isolate specific components of jewelry
- **Attribute Extraction**: Utilizes CLIP-ViT to identify materials, stones, styles, and other attributes
- **Description Generation**: Leverages GPT-4o to create detailed, accurate product descriptions

### Vector Search

- Find similar jewelry items using vector embeddings
- Visual similarity search for customers
- Advanced filtering based on extracted attributes

### Admin Interface

- Dashboard for monitoring AI processing jobs
- Bulk image upload and processing
- Product management with attribute editing
- Category and collection management

### Customer Experience

- Personalized AI styling consultant chatbot
- Interactive product browsing with similar item suggestions
- Secure authentication system

## Architecture

- **Backend**: Python Flask API with PostgreSQL database
- **Frontend**: Next.js with Shadcn UI components and Tailwind CSS
- **AI Processing**: GPU-powered Modal Labs backend
- **Authentication**: Replit Auth integration
- **Containerization**: Docker configuration for easy deployment

## Required Environment Variables

- `DATABASE_URL`: PostgreSQL database connection string
- `MODAL_LABS_API_KEY`: API key for Modal Labs GPU processing
- `OPENAI_API_KEY`: API key for OpenAI services

## Getting Started

### Prerequisites

- Python 3.10+
- Node.js 18+
- PostgreSQL database
- Modal Labs account
- OpenAI API access

### Setup Instructions

1. Clone this repository
2. Set up environment variables (see `.env.example`)
3. Install Python dependencies: `pip install -r requirements.txt`
4. Install Node.js dependencies: `npm install`
5. Start the application: `gunicorn --bind 0.0.0.0:5000 --reuse-port --reload main:app`

## Deployment

The application is containerized for easy deployment to any cloud provider:
- Docker Compose configuration for development
- Production-ready Dockerfiles for API gateway and worker components
- Vercel-compatible Next.js frontend

## License

[Your chosen license]