Title: AI-Powered Jewelry Analysis API & Mobile App

Description: A comprehensive jewelry cataloging and e-commerce platform with advanced AI image analysis capabilities. The application consists of a Python backend API that processes jewelry images through specialized AI models to extract attributes and generate descriptions, paired with a React Native mobile app that provides a seamless shopping experience for jewelry enthusiasts.

Type: Mobile App with Backend API

Platform: iOS/Android (React Native) with Python Backend

Language/Framework: 
- Backend: Python (Flask/FastAPI), PostgreSQL, Modal Labs for GPU compute
- Mobile: React Native with TypeScript, Redux for state management

Key Features:

1. AI Jewelry Analysis Pipeline: Backend API processes jewelry images through specialized models (YOLOv8x, SAM, CLIP-ViT, GPT-4o) to automatically extract product details, with a clean REST API for mobile integration.

2. Visual Search & Recommendations: Mobile app allows users to take photos of jewelry or upload from their gallery, with the backend providing similar items based on visual embeddings and semantic attributes.

3. Admin Mobile Dashboard: Dedicated admin section in the app for inventory management, processing status monitoring, and analytics with real-time updates on sales and customer engagement.