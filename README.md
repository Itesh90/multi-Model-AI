# Multi-Modal AI Platform (Student Edition)

A comprehensive multi-modal AI platform that processes text, images, audio, and video to provide intelligent solutions. Built with a student-friendly approach using modern AI/ML technologies.

## 🚀 Features

- **Multi-Modal Processing**: Text, Image, Audio, and Video analysis
- **Vector Database Integration**: Weaviate for semantic search and similarity
- **Cloud Storage**: Google Cloud Storage for file management
- **RESTful API**: FastAPI-based backend with comprehensive endpoints
- **Real-time Processing**: Async processing with background tasks
- **Comprehensive Testing**: Full test suite with pytest
- **Student-Friendly**: Designed for learning with detailed documentation

## 📋 Prerequisites

- Python 3.9+
- pip package manager
- Git
- Google Cloud Platform account (free tier)
- Weaviate Cloud account (free tier)

## 🛠️ Installation

### Phase 1: Foundation Setup

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "AI Projects/multi Model AI"
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

### Phase 2: Cloud Services Setup

1. **Google Cloud Storage Setup**
   - Create a project at [Google Cloud Console](https://console.cloud.google.com/)
   - Enable Cloud Storage API
   - Create a bucket named `multimodal-student-storage`
   - Download service account key as `gcs-key.json`

2. **Weaviate Cloud Setup**
   - Sign up at [Weaviate Cloud](https://cloud.weaviate.io/)
   - Create a free cluster named `multimodal-student`
   - Note your cluster URL and API key

3. **Update environment variables**
   ```env
   # API Configuration
   API_HOST=localhost
   API_PORT=8000
   DEBUG=True
   
   # Database
   DATABASE_URL=sqlite:///./app.db
   
   # Cloud Storage
   GCS_BUCKET_NAME=multimodal-student-storage
   GCS_KEY_PATH=./gcs-key.json
   
   # Weaviate Vector Database
   WEAVIATE_URL=https://your-cluster.weaviate.network
   WEAVIATE_API_KEY=your-api-key
   
   # AI Model Configuration
   OPENAI_API_KEY=your_openai_api_key
   MODEL_CONFIG_PATH=config/models.json
   ```

## 🏃‍♂️ Quick Start

1. **Initialize the database**
   ```bash
   python backend/database.py
   ```

2. **Start the development server**
   ```bash
   uvicorn backend.main:app --reload
   ```

3. **Access the API documentation**
   - Swagger UI: http://localhost:8000/docs
   - ReDoc: http://localhost:8000/redoc

4. **Test the API**
   ```bash
   # Health check
   curl http://localhost:8000/health
   
   # Process an image (replace with your API key)
   curl -X 'POST' \
     'http://localhost:8000/process-image' \
     -H 'X-API-Key: student-api-key-123' \
     -F 'file=@/path/to/your/test.jpg'
   ```

5. **Run tests**
   ```bash
   pytest
   ```

## 📁 Project Structure

```
multi Model AI/
├── backend/
│   ├── main.py              # FastAPI application entry point
│   ├── database.py          # SQLite database setup and operations
│   ├── vector_db.py         # Weaviate vector database integration
│   ├── storage.py           # Google Cloud Storage operations
│   ├── auth.py              # Authentication and authorization
│   └── services/
│       ├── text_processing.py    # Text analysis and embeddings
│       ├── image_processing.py   # Image analysis and description
│       ├── audio_processing.py   # Audio transcription and analysis
│       └── video_processing.py   # Video analysis and processing
├── docs/
│   └── design.md            # System architecture and design decisions
├── tests/
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── performance/         # Performance tests
├── config/
│   └── models.json          # AI model configurations
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .env.example            # Environment variables template
└── .gitignore              # Git ignore rules
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file with the following variables:

```env
# API Configuration
API_HOST=localhost
API_PORT=8000
DEBUG=True

# Database
DATABASE_URL=sqlite:///./app.db

# Cloud Storage
GCS_BUCKET_NAME=multimodal-student-storage
GCS_KEY_PATH=./gcs-key.json

# Weaviate Vector Database
WEAVIATE_URL=https://your-cluster.weaviate.network
WEAVIATE_API_KEY=your-api-key

# AI Model Configuration
OPENAI_API_KEY=your_openai_api_key
MODEL_CONFIG_PATH=config/models.json

# Security
SECRET_KEY=your-secret-key-here
API_KEY=student-api-key-123
```

### Model Configuration

Create `config/models.json` to configure AI models:

```json
{
  "text": {
    "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
    "sentiment_model": "distilbert-base-uncased-finetuned-sst-2-english",
    "summarization_model": "sshleifer/distilbart-cnn-12-6"
  },
  "image": {
    "description_model": "microsoft/git-base-coco",
    "object_detection_model": "facebook/detr-resnet-50"
  },
  "audio": {
    "transcription_model": "openai/whisper-base",
    "sentiment_model": "facebook/wav2vec2-base"
  }
}
```

## 📚 Documentation

- [Design Documentation](docs/design.md) - System architecture and design decisions
- [API Documentation](http://localhost:8000/docs) - Interactive API documentation
- [Testing Guide](tests/README.md) - Testing strategies and guidelines

## 🧪 Testing

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=backend

# Run specific test categories
pytest tests/unit/
pytest tests/integration/
pytest tests/performance/
```

### Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test system performance under load

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Development Guidelines

- Follow PEP 8 style guidelines
- Write tests for new features
- Update documentation for API changes
- Use type hints in Python code

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the [documentation](docs/)
2. Search existing [issues](../../issues)
3. Create a new issue with detailed information

## 🔮 Development Roadmap

### Phase 1: Foundation & Planning (Weeks 1-3) ✅
- [x] Project structure setup
- [x] Requirements analysis
- [x] Architecture design
- [x] Technology stack selection

### Phase 2: Core Infrastructure (Weeks 4-8) ✅
- [x] Database setup (SQLite + Weaviate)
- [x] Cloud storage integration
- [x] Basic API framework
- [x] Authentication system

### Phase 3: Multi-Modal Processing (Weeks 9-16) 🚧
- [x] Text processing module
- [ ] Image processing module
- [ ] Audio processing module
- [ ] Video processing module
- [ ] Multi-modal fusion

### Phase 4: Advanced Features (Weeks 17-24)
- [ ] RAG (Retrieval-Augmented Generation)
- [ ] Real-time processing
- [ ] Advanced analytics
- [ ] Performance optimization

### Phase 5: Production Deployment (Weeks 25-32)
- [ ] Containerization with Docker
- [ ] CI/CD pipeline
- [ ] Monitoring and logging
- [ ] Production deployment

## 🎓 Student Learning Objectives

This project is designed to teach:

1. **Multi-Modal AI**: Understanding how to process different data types
2. **System Architecture**: Designing scalable AI systems
3. **Cloud Integration**: Working with cloud services and APIs
4. **API Development**: Building RESTful APIs with FastAPI
5. **Vector Databases**: Understanding semantic search and similarity
6. **Testing**: Writing comprehensive tests for AI systems
7. **Deployment**: Taking projects from development to production

## 💡 Tips for Students

- Start with the basic text processing before moving to images/audio
- Use the free tiers of cloud services to avoid costs
- Test each component thoroughly before integrating
- Document your learning process and challenges
- Don't be afraid to ask questions and seek help from the community 