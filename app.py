import streamlit as st
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.document_loaders import DataFrameLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Configuration
st.set_page_config(page_title="Course Search | Analytics Vidhya", layout="wide")

# Custom CSS
st.markdown("""
<style>
    .main { background-color: #f0f2f6; }
    .stButton>button {
        background-color: #3B82F6;
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 0.75rem 1.5rem;
        border: none;
        transition: all 0.3s;
    }
    .stButton>button:hover { background-color: #2563EB; }
    .course-card {
        background-color: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1.5rem;
        transition: transform 0.2s;
    }
    .course-card:hover { transform: translateY(-5px); }
    .metrics-card {
        background: linear-gradient(135deg, #3B82F6, #1E3A8A);
        color: white;
        padding: 1.25rem;
        border-radius: 12px;
        text-align: center;
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.875rem;
        font-weight: 500;
        margin: 0.25rem;
    }
    .search-box {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

def generate_extended_course_data():
    categories = {
        "Programming": {
            "topics": ["Python", "Java", "JavaScript", "C++", "Ruby", "Go", "Rust", "PHP", "Swift", "Kotlin", 
                      "TypeScript", "Scala", "R", "MATLAB", "Dart", "Perl", "Shell Scripting", "Assembly", "Haskell", "Lua"],
            "specializations": ["Web", "Mobile", "Game", "Systems", "Database"],
            "frameworks": ["Spring", "Django", "React", "Angular", "Vue"],
            "base_price": 99.99
        },
        "Machine Learning": {
            "topics": ["Neural Networks", "Deep Learning", "NLP", "Computer Vision", "Reinforcement Learning", 
                      "GANs", "Transfer Learning", "AutoML", "Feature Engineering", "Model Deployment",
                      "Quantum ML", "Edge AI", "Federated Learning", "Meta Learning", "Active Learning",
                      "Few-Shot Learning", "Anomaly Detection", "Time Series", "Graph Neural Networks", "MLOps"],
            "specializations": ["Research", "Industry", "Healthcare", "Finance", "Robotics"],
            "frameworks": ["TensorFlow", "PyTorch", "Keras", "Scikit-learn", "JAX"],
            "base_price": 149.99
        },
        "Data Science": {
            "topics": ["Statistics", "Data Analysis", "Data Visualization", "Big Data", "Data Mining",
                      "Business Intelligence", "Data Engineering", "Data Governance", "Data Strategy", "Analytics",
                      "Experimental Design", "Causal Inference", "A/B Testing", "Data Ethics", "Data Quality",
                      "Market Research", "Customer Analytics", "Risk Analytics", "Social Network Analysis", "Text Mining"],
            "specializations": ["Business", "Science", "Marketing", "Finance", "Healthcare"],
            "frameworks": ["Pandas", "Spark", "Hadoop", "Tableau", "Power BI"],
            "base_price": 129.99
        },
        "Cloud Computing": {
            "topics": ["AWS", "Azure", "Google Cloud", "DevOps", "Kubernetes", "Docker", "Microservices", "Serverless",
                      "Cloud Security", "IaC", "Edge Computing", "Multi-Cloud", "Hybrid Cloud", "Cloud Migration",
                      "Cloud Architecture", "Cloud Optimization", "Service Mesh", "Cloud Native", "Containers", "CI/CD"],
            "specializations": ["Architecture", "Development", "Security", "Operations", "Management"],
            "frameworks": ["Terraform", "Ansible", "Jenkins", "GitOps", "ArgoCD"],
            "base_price": 139.99
        },
        "Web Development": {
            "topics": ["HTML/CSS", "JavaScript", "React", "Angular", "Vue.js", "Node.js", "Express", "Django",
                      "Flask", "Spring", "GraphQL", "WebAssembly", "Progressive Web Apps", "Web Security",
                      "Web Performance", "Responsive Design", "Web Accessibility", "SEO", "Web Analytics", "Web3"],
            "specializations": ["Frontend", "Backend", "Full-Stack", "E-commerce", "CMS"],
            "frameworks": ["Next.js", "Nuxt.js", "Svelte", "Laravel", "Ruby on Rails"],
            "base_price": 89.99
        }
    }

    levels = ["Beginner", "Intermediate", "Advanced"]
    prefixes = ["Complete Guide to", "Professional", "Mastering", "Advanced", "Expert", 
                "Practical", "Essential", "Modern", "Comprehensive", "Ultimate"]
    suffixes = ["Bootcamp", "Masterclass", "Certification", "Deep Dive", "Workshop"]

    courses = []
    for category, info in categories.items():
        for topic in info["topics"]:
            for level in levels:
                for spec in info["specializations"]:
                    for framework in info["frameworks"]:
                        for prefix in prefixes:
                            for suffix in suffixes:
                                title = f"{prefix} {topic} for {spec} with {framework} - {suffix}"
                                description = (f"Master {topic} in {spec} using {framework}. Industry-standard "
                                            f"{level.lower()}-level training with hands-on projects.")
                                
                                base_duration = {"Beginner": 6, "Intermediate": 8, "Advanced": 12}[level]
                                duration = base_duration + (suffixes.index(suffix) * 2)
                                
                                price_modifier = {"Beginner": 1.0, "Intermediate": 1.3, "Advanced": 1.5}[level]
                                price_modifier += suffixes.index(suffix) * 0.1
                                
                                curriculum = (f"{level} {topic}, {spec} applications, {framework} development, "
                                           f"Industry tools, Real-world projects, Career guidance")

                                course = {
                                    "title": title,
                                    "description": description,
                                    "curriculum": curriculum,
                                    "category": category,
                                    "difficulty": level,
                                    "duration_weeks": duration,
                                    "price": round(info["base_price"] * price_modifier, 2)
                                }
                                courses.append(course)
                                
                                if len(courses) >= 100:  # Take only first 100 courses per category
                                    break
                            if len(courses) >= 100:
                                break
                        if len(courses) >= 100:
                            break
                    if len(courses) >= 100:
                        break
                if len(courses) >= 100:
                    break
            if len(courses) >= 100:
                break
    
    return courses[:100]  # Ensure exactly 100 courses per category

# Usage in your main application
def _load_course_data(self) -> pd.DataFrame:
    all_courses = []
    for _ in range(5):  # Generate 100 courses for each of the 5 categories
        courses = generate_extended_course_data()
        all_courses.extend(courses)
    return pd.DataFrame(all_courses)
class CourseSearch:
    def __init__(self):
        self.vectorstore = None
        self.df = self.load_course_data()
        self.initialize_search()

    def load_course_data(self):
        return pd.DataFrame({
            'title': [
                'Python for Data Science', 'Machine Learning Basics',
                'Deep Learning Fundamentals', 'SQL Masterclass',
                'Data Visualization'  # Add more courses
            ],
            'description': [
                'Comprehensive Python programming for data analysis',
                'Introduction to machine learning algorithms',
                'Neural networks and deep learning basics',
                'Advanced SQL for data analysis',
                'Data visualization techniques'  # Add more descriptions
            ],
            'curriculum': [
                'Python basics, Pandas, NumPy',
                'Supervised learning, Classification',
                'Neural networks, PyTorch',
                'SQL queries, Database design',
                'Matplotlib, Seaborn, Plotly'  # Add more curricula
            ],
            'category': [
                'Programming', 'Machine Learning',
                'Deep Learning', 'Database',
                'Visualization'  # Add more categories
            ]
        })

    def initialize_search(self):
        self.df['content'] = self.df.apply(
            lambda x: f"Title: {x['title']}\nDescription: {x['description']}\nCurriculum: {x['curriculum']}", 
            axis=1
        )
        loader = DataFrameLoader(self.df, page_content_column="content")
        documents = loader.load()
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = text_splitter.split_documents(documents)
        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
        self.vectorstore = FAISS.from_documents(split_docs, embeddings)

    def search(self, query):
        if not query:
            return []
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            huggingfacehub_api_token="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"
        )
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=self.vectorstore.as_retriever(search_kwargs={"k": 5})
        )
        return qa_chain.run(query)

def main():
    search_system = CourseSearch()
    
    # Sidebar
    with st.sidebar:
        st.markdown("### Platform Analytics")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                <div class="metrics-card">
                    <h3>Total Courses</h3>
                    <h2>100+</h2>
                </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown("""
                <div class="metrics-card">
                    <h3>Categories</h3>
                    <h2>5</h2>
                </div>
            """, unsafe_allow_html=True)
        
        # Category distribution
        fig = px.pie(
            search_system.df,
            names='category',
            title='Course Distribution',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig, use_container_width=True)

    # Main content
    st.title("🎓 Analytics Vidhya Course Search")
    st.markdown("### Discover Your Perfect Learning Path")

    # Search interface
    with st.container():
        col1, col2 = st.columns([2,1])
        with col1:
            query = st.text_input("🔍 Search courses:",
                placeholder="E.g., 'Python for beginners' or 'deep learning'")
        with col2:
            category = st.selectbox("Category",
                ["All"] + list(search_system.df['category'].unique()))

    if st.button("🔍 Search", type="primary"):
        if query:
            with st.spinner("Searching..."):
                results = search_system.search(query)
                st.markdown("### 🎯 Search Results")
                st.markdown(f"""
                    <div class="course-card">
                        <h4>{results}</h4>
                    </div>
                """, unsafe_allow_html=True)

                # Related courses
                st.markdown("### 📚 Related Courses")
                filtered_df = search_system.df
                if category != "All":
                    filtered_df = filtered_df[filtered_df['category'] == category]
                
                for _, course in filtered_df.head(3).iterrows():
                    st.markdown(f"""
                        <div class="course-card">
                            <h4>{course['title']}</h4>
                            <p>{course['description']}</p>
                            <div class="badge" style="background: #DBEAFE; color: #1E40AF">
                                {course['category']}
                            </div>
                            <p><strong>Curriculum:</strong> {course['curriculum']}</p>
                        </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("### ⭐ Featured Courses")
        for _, course in search_system.df.head(5).iterrows():
            st.markdown(f"""
                <div class="course-card">
                    <h4>{course['title']}</h4>
                    <p>{course['description']}</p>
                    <div class="badge" style="background: #DBEAFE; color: #1E40AF">
                        {course['category']}
                    </div>
                    <p><strong>Curriculum:</strong> {course['curriculum']}</p>
                </div>
            """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()