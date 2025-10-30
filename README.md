# 📍 Memory Map with AI Search

Interactive memory map application with AI-powered search using modular agentic architecture. Features natural language queries, visual photo mapping, and semantic similarity search through coordinated Perception, Memory, Decision, and Action modules.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🌟 Features

- **🔍 Natural Language Search**: Ask questions like "Show my mountain trips" or "Where did I click food photos?"
- **🗺️ Interactive Map**: Visual representation of memories with location pins
- **🖼️ Photo Integration**: Display photos with captions in interactive popups
- **🎯 Smart Highlighting**: Top 3 matching results highlighted in red on the map
- **🧠 AI-Powered**: Uses SentenceTransformers for semantic similarity search
- **⚡ Modular Architecture**: Clean separation of concerns with agentic design

## 🏗️ Architecture

The application follows a modular agentic architecture with five core components:

```
📁 memory_map/
├── 🌐 app.py           # Main Streamlit UI
├── 🧠 agent.py         # Orchestrator (coordinates all modules)
├── 👁️ perception.py    # Query understanding & embedding
├── 💾 memory.py        # Data storage & retrieval
├── 🧭 decision.py      # Reasoning & decision making
├── ⚡ action.py        # Visual display & execution
├── 📊 memories.csv     # Memory data
└── 🖼️ images/          # Photo storage
```

### Module Responsibilities

| Module | Role | Description |
|--------|------|-------------|
| **🧠 Agent** | Orchestrator | Coordinates all modules and manages the query pipeline |
| **👁️ Perception** | Input Processing | Converts text queries into embeddings using SentenceTransformers |
| **💾 Memory** | Data Management | Handles CSV data, FAISS indexing, and similarity search |
| **🧭 Decision** | Reasoning | Decides what actions to take based on retrieved results |
| **⚡ Action** | Execution | Creates maps, popups, and visual displays using Folium |

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- pip or conda package manager

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/RoyRushreeta/memory-map.git
   cd memory-map
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare your data**
   - Add your photos to the `images/` folder
   - Update `memories.csv` with your memory data:
   ```csv
   location,latitude,longitude,caption,image
   Goa,15.6745,73.7068,Sunset at Arambol Beach,IMG_20231025_121448.jpg
   Bangalore,13.0411,77.6153,Perfect cloudy sky,IMG_20251005_110948.jpg
   ```

4. **Run the application**
   ```bash
   streamlit run app.py
   ```

5. **Open your browser** and navigate to `http://localhost:8501`

## 📊 Data Format

The `memories.csv` file should contain the following columns:

| Column | Type | Description | Example |
|--------|------|-------------|---------|
| `location` | String | Place name | "Goa" |
| `latitude` | Float | GPS latitude | 15.6745 |
| `longitude` | Float | GPS longitude | 73.7068 |
| `caption` | String | Memory description | "Sunset at Arambol Beach" |
| `image` | String | Image filename | "IMG_20231025_121448.jpg" |

## 🎮 Usage

### Basic Interaction

1. **View All Memories**: Load the app to see all memories on the map
2. **Search Memories**: Enter natural language queries in the search box
3. **Explore Results**: 
   - Top 3 matches highlighted in red
   - Other memories shown in blue
   - Click markers to see photos and details

### Example Queries

- `"Show my mountain trips"`
- `"Where did I click food photos?"`
- `"Sunset photos with family"`
- `"Recent vacation memories"`
- `"Beautiful landscape shots"`

## 🔧 Configuration

### Image Settings

Modify image processing parameters in `action.py`:

```python
action.set_image_settings(
    max_width=400,      # Maximum image width
    max_height=300,     # Maximum image height
    quality=85          # JPEG quality (1-100)
)
```

### Search Parameters

Adjust search behavior in `decision.py`:

```python
decision.set_similarity_threshold(0.3)  # Minimum similarity score
decision.set_default_top_n(3)          # Number of top results
```

## 🧪 Testing

Test individual modules:

```bash
# Test Agent initialization
python -c "from agent import Agent; agent = Agent(); print('✅ Agent ready!')"

# Test search functionality
python -c "
from agent import Agent
agent = Agent()
results, scores = agent.search_memories('sunset photos')
print(f'Found {len(results)} results')
"
```

## 📁 Project Structure

```
memory_map/
│
├── app.py                 # 🌐 Main Streamlit application
├── agent.py              # 🧠 Main orchestrator class
├── perception.py         # 👁️ Query processing & embeddings
├── memory.py             # 💾 Data loading & FAISS search
├── decision.py           # 🧭 Decision making logic
├── action.py             # ⚡ Visual display & Folium maps
├── memories.csv          # 📊 Memory data file
├── requirements.txt      # 📦 Python dependencies
├── README.md            # 📖 This file
└── images/              # 🖼️ Photo storage directory
    ├── IMG_001.jpg
    ├── IMG_002.jpg
    └── ...
```

## 🛠️ Development

### Adding New Features

1. **New Query Types**: Extend `decision.py` with additional intent analysis
2. **Enhanced Visuals**: Modify `action.py` for new map features
3. **Better Search**: Improve embedding models in `perception.py`
4. **Data Sources**: Extend `memory.py` for new data formats

### Code Quality

The codebase follows these principles:
- **Modular Design**: Each module has a single responsibility
- **Clean Interfaces**: Well-defined APIs between modules
- **Documentation**: Comprehensive docstrings and comments
- **Error Handling**: Graceful failure management
- **Caching**: Streamlit caching for performance

## 📚 Dependencies

### Core Dependencies

- **streamlit**: Web app framework
- **folium**: Interactive map generation
- **streamlit-folium**: Streamlit-Folium integration
- **sentence-transformers**: Text embedding models
- **faiss-cpu**: Efficient similarity search
- **pandas**: Data manipulation
- **PIL (Pillow)**: Image processing
- **numpy**: Numerical operations

### Optional Dependencies

- **matplotlib**: Additional visualization (if needed)
- **seaborn**: Statistical plotting (if needed)

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔮 Future Enhancements

- [ ] **Multi-language Support**: Support for queries in different languages
- [ ] **Advanced Filters**: Date range, location radius filtering
- [ ] **Batch Upload**: Easy bulk photo and data import
- [ ] **Export Features**: Export maps and results
- [ ] **Mobile Optimization**: Responsive design for mobile devices
- [ ] **Cloud Storage**: Integration with Google Photos, Dropbox
- [ ] **Social Features**: Share memories and maps

## 📞 Support

If you encounter any issues or have questions:

1. Check the [Issues](https://github.com/RoyRushreeta/memory-map/issues) page
2. Create a new issue with detailed description
3. Include error messages and system information

## 🙏 Acknowledgments

- **Streamlit**: For the amazing web app framework
- **Folium**: For beautiful interactive maps
- **SentenceTransformers**: For powerful text embeddings
- **FAISS**: For efficient similarity search
- **Open Source Community**: For inspiration and tools

---

**Made with ❤️ by [RoyRushreeta](https://github.com/RoyRushreeta)**

*Transform your memories into an interactive, searchable map experience!*