import streamlit as st
import plotly.express as px
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import os
from typing import List, Dict
import pandas as pd
from difflib import SequenceMatcher

# Set your Hugging Face API token
os.environ["HUGGING_FACE_TOKEN"] = "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"

# Initialize the model
@st.cache_resource
def load_model():
    return SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

course_database = {
    'courses': [
        # Business Analytics Courses (10)
        {
            'id': 'BA001',
            'title': 'Business Analytics Fundamentals',
            'category': 'Business Analytics',
            'subcategory': 'Fundamentals',
            'tags': ['analytics', 'business intelligence', 'data-driven decisions', 'KPI', 'metrics'],
            'level': 'Beginner',
            'duration': '4 weeks',
            'is_free': True,
            'description': 'Learn the basics of business analytics and how to make data-driven decisions',
            'curriculum': ['Introduction to Business Analytics', 'Data Analysis Tools', 'KPI Definition', 'Reporting'],
            'prerequisites': ['Basic Excel'],
            'skills_gained': ['Data Analysis', 'Business Intelligence', 'Report Creation'],
            'tools_used': ['Excel', 'Tableau', 'Power BI'],
            'image': '📊'
        },
        {
            'id': 'BA002',
            'title': 'Advanced Business Intelligence',
            'category': 'Business Analytics',
            'subcategory': 'Advanced',
            'tags': ['data visualization', 'predictive analytics', 'statistical analysis', 'forecasting'],
            'level': 'Advanced',
            'duration': '8 weeks',
            'is_free': False,
            'description': 'Master advanced business intelligence techniques and predictive analytics',
            'curriculum': ['Advanced Data Visualization', 'Predictive Modeling', 'Time Series Analysis', 'Business Forecasting'],
            'prerequisites': ['Basic Statistics', 'SQL', 'Business Analytics Fundamentals'],
            'skills_gained': ['Predictive Analytics', 'Advanced Visualization', 'Statistical Modeling'],
            'tools_used': ['Python', 'R', 'Tableau', 'Power BI'],
            'image': '📈'
        },
        {
            'id': 'BA003',
            'title': 'Financial Analytics',
            'category': 'Business Analytics',
            'subcategory': 'Finance',
            'tags': ['financial analysis', 'risk assessment', 'portfolio management', 'financial forecasting'],
            'level': 'Intermediate',
            'duration': '6 weeks',
            'is_free': False,
            'description': 'Apply analytics techniques to financial data and decision-making',
            'curriculum': ['Financial Metrics', 'Risk Analysis', 'Portfolio Analytics', 'Financial Modeling'],
            'prerequisites': ['Business Analytics Fundamentals', 'Basic Finance'],
            'skills_gained': ['Financial Modeling', 'Risk Assessment', 'Portfolio Analysis'],
            'tools_used': ['Excel', 'R', 'Bloomberg Terminal'],
            'image': '💹'
        },
        {
            'id': 'BA004',
            'title': 'Marketing Analytics',
            'category': 'Business Analytics',
            'subcategory': 'Marketing',
            'tags': ['marketing', 'customer analytics', 'campaign analysis', 'ROI tracking'],
            'level': 'Intermediate',
            'duration': '6 weeks',
            'is_free': False,
            'description': 'Learn to analyze marketing campaigns and customer behavior',
            'curriculum': ['Customer Segmentation', 'Campaign Analytics', 'ROI Analysis', 'Customer Journey Analysis'],
            'prerequisites': ['Basic Marketing', 'Business Analytics Fundamentals'],
            'skills_gained': ['Marketing Analytics', 'Customer Analysis', 'Campaign Optimization'],
            'tools_used': ['Google Analytics', 'Excel', 'Mixpanel'],
            'image': '📣'
        },
        {
            'id': 'BA005',
            'title': 'HR Analytics',
            'category': 'Business Analytics',
            'subcategory': 'Human Resources',
            'tags': ['HR', 'workforce analytics', 'talent management', 'employee metrics'],
            'level': 'Intermediate',
            'duration': '4 weeks',
            'is_free': True,
            'description': 'Master HR analytics for better workforce management',
            'curriculum': ['Workforce Metrics', 'Talent Analytics', 'Retention Analysis', 'Performance Analytics'],
            'prerequisites': ['HR Basics', 'Business Analytics Fundamentals'],
            'skills_gained': ['HR Metrics', 'Workforce Analysis', 'Retention Modeling'],
            'tools_used': ['Excel', 'Tableau', 'SPSS'],
            'image': '👥'
        },
        {
            'id': 'BA006',
            'title': 'Supply Chain Analytics',
            'category': 'Business Analytics',
            'subcategory': 'Supply Chain',
            'tags': ['supply chain', 'logistics', 'inventory management', 'optimization'],
            'level': 'Advanced',
            'duration': '8 weeks',
            'is_free': False,
            'description': 'Apply analytics to optimize supply chain operations',
            'curriculum': ['Supply Chain Metrics', 'Inventory Analytics', 'Logistics Optimization', 'Network Analysis'],
            'prerequisites': ['Supply Chain Basics', 'Advanced Business Analytics'],
            'skills_gained': ['Supply Chain Optimization', 'Logistics Analysis', 'Network Design'],
            'tools_used': ['Python', 'SAP', 'Tableau'],
            'image': '🏭'
        },
        {
            'id': 'BA007',
            'title': 'Customer Analytics',
            'category': 'Business Analytics',
            'subcategory': 'Customer Insights',
            'tags': ['customer behavior', 'CLV', 'churn analysis', 'customer segmentation'],
            'level': 'Intermediate',
            'duration': '6 weeks',
            'is_free': False,
            'description': 'Understand customer behavior through data analysis',
            'curriculum': ['Customer Lifetime Value', 'Churn Prediction', 'Behavioral Analysis', 'Segmentation'],
            'prerequisites': ['Business Analytics Fundamentals', 'Basic Statistics'],
            'skills_gained': ['Customer Analysis', 'Behavioral Analytics', 'Churn Prevention'],
            'tools_used': ['Python', 'R', 'Salesforce'],
            'image': '👥'
        },
        {
            'id': 'BA008',
            'title': 'Operations Analytics',
            'category': 'Business Analytics',
            'subcategory': 'Operations',
            'tags': ['operations', 'process optimization', 'efficiency metrics', 'quality control'],
            'level': 'Advanced',
            'duration': '8 weeks',
            'is_free': False,
            'description': 'Optimize business operations through data analysis',
            'curriculum': ['Process Analytics', 'Quality Metrics', 'Efficiency Analysis', 'Operations Optimization'],
            'prerequisites': ['Operations Management', 'Advanced Analytics'],
            'skills_gained': ['Process Optimization', 'Quality Analysis', 'Efficiency Improvement'],
            'tools_used': ['Python', 'Tableau', 'Six Sigma Tools'],
            'image': '⚙️'
        },
        {
            'id': 'BA009',
            'title': 'Real Estate Analytics',
            'category': 'Business Analytics',
            'subcategory': 'Real Estate',
            'tags': ['real estate', 'property analysis', 'market trends', 'valuation'],
            'level': 'Intermediate',
            'duration': '6 weeks',
            'is_free': False,
            'description': 'Apply analytics to real estate market analysis',
            'curriculum': ['Market Analysis', 'Property Valuation', 'Location Analytics', 'Investment Analysis'],
            'prerequisites': ['Real Estate Basics', 'Business Analytics'],
            'skills_gained': ['Market Analysis', 'Property Valuation', 'Investment Analytics'],
            'tools_used': ['Excel', 'GIS Software', 'Tableau'],
            'image': '🏘️'
        },
        {
            'id': 'BA010',
            'title': 'Healthcare Analytics',
            'category': 'Business Analytics',
            'subcategory': 'Healthcare',
            'tags': ['healthcare', 'patient analytics', 'clinical metrics', 'health outcomes'],
            'level': 'Advanced',
            'duration': '10 weeks',
            'is_free': False,
            'description': 'Master healthcare analytics and patient outcome analysis',
            'curriculum': ['Clinical Analytics', 'Patient Metrics', 'Outcome Analysis', 'Healthcare Operations'],
            'prerequisites': ['Healthcare Background', 'Advanced Analytics'],
            'skills_gained': ['Healthcare Metrics', 'Patient Analysis', 'Outcome Measurement'],
            'tools_used': ['R', 'SPSS', 'Healthcare BI Tools'],
            'image': '🏥'
        },

        # Generative AI Courses (10)
        {
            'id': 'GEN001',
            'title': 'Introduction to Generative AI',
            'category': 'Generative AI',
            'subcategory': 'Fundamentals',
            'tags': ['generative AI', 'LLMs', 'foundations', 'AI basics'],
            'level': 'Beginner',
            'duration': '4 weeks',
            'is_free': True,
            'description': 'Learn the fundamentals of generative AI and its applications',
            'curriculum': ['AI Basics', 'Types of Generative Models', 'Applications', 'Ethics'],
            'prerequisites': ['Basic Programming'],
            'skills_gained': ['AI Fundamentals', 'Model Understanding', 'Application Design'],
            'tools_used': ['Python', 'Jupyter', 'Hugging Face'],
            'image': '🤖'
        },
        {
            'id': 'GEN002',
            'title': 'Large Language Models',
            'category': 'Generative AI',
            'subcategory': 'LLMs',
            'tags': ['LLM', 'NLP', 'transformer models', 'language AI'],
            'level': 'Intermediate',
            'duration': '8 weeks',
            'is_free': False,
            'description': 'Deep dive into Large Language Models and their applications',
            'curriculum': ['Transformer Architecture', 'LLM Training', 'Fine-tuning', 'Deployment'],
            'prerequisites': ['Python', 'Deep Learning Basics'],
            'skills_gained': ['LLM Development', 'Model Training', 'Application Building'],
            'tools_used': ['PyTorch', 'Transformers', 'TensorFlow'],
            'image': '📚'
        },
        {
            'id': 'GEN003',
            'title': 'Prompt Engineering',
            'category': 'Generative AI',
            'subcategory': 'Prompting',
            'tags': ['prompt engineering', 'LLM interaction', 'AI communication'],
            'level': 'Beginner',
            'duration': '4 weeks',
            'is_free': True,
            'description': 'Master the art of prompt engineering for AI models',
            'curriculum': ['Prompt Basics', 'Advanced Techniques', 'Use Cases', 'Best Practices'],
            'prerequisites': ['None'],
            'skills_gained': ['Prompt Design', 'AI Interaction', 'Output Optimization'],
            'tools_used': ['ChatGPT', 'Claude', 'GPT Tools'],
            'image': '✍️'
        },
        {
            'id': 'GEN004',
            'title': 'AI Image Generation',
            'category': 'Generative AI',
            'subcategory': 'Image Generation',
            'tags': ['image generation', 'DALL-E', 'Stable Diffusion', 'visual AI'],
            'level': 'Intermediate',
            'duration': '6 weeks',
            'is_free': False,
            'description': 'Learn to create and manipulate images using AI',
            'curriculum': ['Image Models', 'Generation Techniques', 'Style Transfer', 'Applications'],
            'prerequisites': ['Basic Python', 'Deep Learning Concepts'],
            'skills_gained': ['Image Generation', 'Model Usage', 'Creative Applications'],
            'tools_used': ['Stable Diffusion', 'DALL-E', 'Midjourney'],
            'image': '🎨'
        },
        {
            'id': 'GEN005',
            'title': 'Generative AI Ethics',
            'category': 'Generative AI',
            'subcategory': 'Ethics',
            'tags': ['AI ethics', 'responsible AI', 'bias', 'governance'],
            'level': 'Intermediate',
            'duration': '4 weeks',
            'is_free': True,
            'description': 'Explore ethical considerations in generative AI',
            'curriculum': ['Ethical Framework', 'Bias Detection', 'Governance', 'Best Practices'],
            'prerequisites': ['Basic AI Knowledge'],
            'skills_gained': ['Ethical Analysis', 'Bias Recognition', 'Responsible Development'],
            'tools_used': ['Analysis Tools', 'Bias Detection Tools'],
            'image': '⚖️'
        },
        {
            'id': 'GEN006',
            'title': 'AI Audio Generation',
            'category': 'Generative AI',
            'subcategory': 'Audio',
            'tags': ['audio AI', 'speech synthesis', 'music generation'],
            'level': 'Advanced',
            'duration': '8 weeks',
            'is_free': False,
            'description': 'Create and manipulate audio using AI techniques',
            'curriculum': ['Audio Models', 'Speech Synthesis', 'Music Generation', 'Applications'],
            'prerequisites': ['Python', 'Signal Processing'],
            'skills_gained': ['Audio Generation', 'Speech Synthesis', 'Music Creation'],
            'tools_used': ['TensorFlow', 'PyDub', 'Librosa'],
            'image': '🎵'
        },
        {
            'id': 'GEN007',
            'title': 'AI Content Creation',
            'category': 'Generative AI',
            'subcategory': 'Content',
            'tags': ['content generation', 'marketing', 'copywriting'],
            'level': 'Beginner',
            'duration': '6 weeks',
            'is_free': False,
            'description': 'Learn to create content using AI tools',
            'curriculum': ['Content Strategy', 'Tool Selection', 'Quality Control', 'Workflow'],
            'prerequisites': ['Basic Writing Skills'],
            'skills_gained': ['Content Generation', 'Tool Mastery', 'Quality Assurance'],
            'tools_used': ['GPT-4', 'Jasper', 'Copy.ai'],
            'image': '📝'
        },
        {
            'id': 'GEN008',
            'title': 'AI Model Fine-tuning',
            'category': 'Generative AI',
            'subcategory': 'Development',
            'tags': ['fine-tuning', 'model adaptation', 'customization'],
            'level': 'Advanced',
            'duration': '10 weeks',
            'is_free': False,
            'description': 'Master the art of fine-tuning AI models',
            'curriculum': ['Fine-tuning Basics', 'Advanced Techniques', 'Evaluation', 'Deployment'],
            'prerequisites': ['Deep Learning', 'Python'],
            'skills_gained': ['Model Fine-tuning', 'Evaluation', 'Deployment'],
            'tools_used': ['PyTorch', 'TensorFlow', 'Hugging Face'],
            'image': '🔧'
        },
        # Continuing Generative AI courses
        {
            'id': 'GEN009',
            'title': 'AI Video Generation',
            'category': 'Generative AI',
            'subcategory': 'Video',
            'tags': ['video generation', 'animation', 'motion synthesis', 'video AI'],
            'level': 'Advanced',
            'duration': '8 weeks',
            'is_free': False,
            'description': 'Learn to generate and manipulate videos using AI techniques',
            'curriculum': ['Video Generation Models', 'Motion Synthesis', 'Style Transfer', 'Video Editing'],
            'prerequisites': ['Deep Learning', 'Computer Vision'],
            'skills_gained': ['Video Generation', 'Motion Synthesis', 'AI Video Editing'],
            'tools_used': ['PyTorch', 'OpenCV', 'RunwayML'],
            'image': '🎥'
        },
        {
            'id': 'GEN010',
            'title': 'AI for Game Development',
            'category': 'Generative AI',
            'subcategory': 'Gaming',
            'tags': ['game development', 'procedural generation', 'AI NPCs', 'game design'],
            'level': 'Advanced',
            'duration': '10 weeks',
            'is_free': False,
            'description': 'Apply generative AI techniques to game development',
            'curriculum': ['Procedural Content Generation', 'NPC Behavior', 'Level Design', 'Asset Creation'],
            'prerequisites': ['Game Development Basics', 'Python'],
            'skills_gained': ['AI Game Development', 'Procedural Generation', 'NPC Design'],
            'tools_used': ['Unity', 'Unreal Engine', 'TensorFlow'],
            'image': '🎮'
        },

        # Machine Learning Courses (10)
        {
            'id': 'ML001',
            'title': 'Machine Learning Fundamentals',
            'category': 'Machine Learning',
            'subcategory': 'Fundamentals',
            'tags': ['ML basics', 'algorithms', 'data science', 'supervised learning'],
            'level': 'Beginner',
            'duration': '6 weeks',
            'is_free': True,
            'description': 'Introduction to core machine learning concepts and algorithms',
            'curriculum': ['ML Basics', 'Supervised Learning', 'Unsupervised Learning', 'Model Evaluation'],
            'prerequisites': ['Basic Python', 'Mathematics'],
            'skills_gained': ['ML Algorithms', 'Data Analysis', 'Model Building'],
            'tools_used': ['scikit-learn', 'Python', 'Jupyter'],
            'image': '🤖'
        },
        {
            'id': 'ML002',
            'title': 'Advanced Machine Learning',
            'category': 'Machine Learning',
            'subcategory': 'Advanced',
            'tags': ['advanced ML', 'optimization', 'ensemble methods', 'feature engineering'],
            'level': 'Advanced',
            'duration': '10 weeks',
            'is_free': False,
            'description': 'Deep dive into advanced machine learning techniques',
            'curriculum': ['Advanced Algorithms', 'Ensemble Methods', 'Feature Engineering', 'Model Optimization'],
            'prerequisites': ['ML Fundamentals', 'Statistics'],
            'skills_gained': ['Advanced ML', 'Model Optimization', 'Feature Engineering'],
            'tools_used': ['Python', 'R', 'TensorFlow'],
            'image': '📊'
        },
        {
            'id': 'ML003',
            'title': 'Time Series Analysis',
            'category': 'Machine Learning',
            'subcategory': 'Time Series',
            'tags': ['time series', 'forecasting', 'sequential data', 'prediction'],
            'level': 'Intermediate',
            'duration': '8 weeks',
            'is_free': False,
            'description': 'Master time series analysis and forecasting techniques',
            'curriculum': ['Time Series Basics', 'Forecasting Methods', 'ARIMA Models', 'Deep Learning for Time Series'],
            'prerequisites': ['ML Fundamentals', 'Statistics'],
            'skills_gained': ['Time Series Analysis', 'Forecasting', 'Sequential Modeling'],
            'tools_used': ['Python', 'statsmodels', 'Prophet'],
            'image': '📈'
        },
        {
            'id': 'ML004',
            'title': 'Reinforcement Learning',
            'category': 'Machine Learning',
            'subcategory': 'RL',
            'tags': ['reinforcement learning', 'AI agents', 'Q-learning', 'policy optimization'],
            'level': 'Advanced',
            'duration': '10 weeks',
            'is_free': False,
            'description': 'Learn reinforcement learning from theory to practice',
            'curriculum': ['RL Fundamentals', 'Q-Learning', 'Policy Gradient Methods', 'Deep RL'],
            'prerequisites': ['Advanced ML', 'Python'],
            'skills_gained': ['RL Algorithms', 'Agent Design', 'Policy Optimization'],
            'tools_used': ['OpenAI Gym', 'PyTorch', 'TensorFlow'],
            'image': '🎮'
        },
        {
            'id': 'ML005',
            'title': 'Machine Learning for Finance',
            'category': 'Machine Learning',
            'subcategory': 'Finance',
            'tags': ['financial ML', 'trading', 'risk analysis', 'portfolio optimization'],
            'level': 'Intermediate',
            'duration': '8 weeks',
            'is_free': False,
            'description': 'Apply machine learning to financial markets and analysis',
            'curriculum': ['Financial Data Analysis', 'Trading Strategies', 'Risk Modeling', 'Portfolio Optimization'],
            'prerequisites': ['ML Fundamentals', 'Finance Basics'],
            'skills_gained': ['Financial ML', 'Trading Strategies', 'Risk Analysis'],
            'tools_used': ['Python', 'Pandas', 'scikit-learn'],
            'image': '💹'
        },
        {
            'id': 'ML006',
            'title': 'Anomaly Detection',
            'category': 'Machine Learning',
            'subcategory': 'Anomaly Detection',
            'tags': ['anomaly detection', 'fraud detection', 'outlier analysis', 'security'],
            'level': 'Intermediate',
            'duration': '6 weeks',
            'is_free': False,
            'description': 'Master techniques for detecting anomalies and outliers',
            'curriculum': ['Statistical Methods', 'ML-based Detection', 'Real-time Analysis', 'Applications'],
            'prerequisites': ['ML Fundamentals', 'Statistics'],
            'skills_gained': ['Anomaly Detection', 'Fraud Analysis', 'Security Analytics'],
            'tools_used': ['Python', 'scikit-learn', 'PyOD'],
            'image': '🔍'
        },
        {
            'id': 'ML007',
            'title': 'Recommendation Systems',
            'category': 'Machine Learning',
            'subcategory': 'RecSys',
            'tags': ['recommendations', 'collaborative filtering', 'content-based filtering'],
            'level': 'Intermediate',
            'duration': '8 weeks',
            'is_free': False,
            'description': 'Build modern recommendation systems using ML',
            'curriculum': ['RecSys Basics', 'Collaborative Filtering', 'Content-Based Systems', 'Hybrid Methods'],
            'prerequisites': ['ML Fundamentals', 'Python'],
            'skills_gained': ['RecSys Development', 'User Modeling', 'System Design'],
            'tools_used': ['Python', 'Surprise', 'TensorFlow'],
            'image': '📱'
        },
        {
            'id': 'ML008',
            'title': 'MLOps Fundamentals',
            'category': 'Machine Learning',
            'subcategory': 'Operations',
            'tags': ['MLOps', 'deployment', 'monitoring', 'infrastructure'],
            'level': 'Advanced',
            'duration': '10 weeks',
            'is_free': False,
            'description': 'Learn to deploy and maintain ML systems in production',
            'curriculum': ['ML Pipeline Design', 'Deployment Strategies', 'Monitoring', 'Infrastructure'],
            'prerequisites': ['ML Experience', 'Software Engineering'],
            'skills_gained': ['MLOps', 'Deployment', 'System Design'],
            'tools_used': ['Docker', 'Kubernetes', 'MLflow'],
            'image': '⚙️'
        },
        {
            'id': 'ML009',
            'title': 'Healthcare Machine Learning',
            'category': 'Machine Learning',
            'subcategory': 'Healthcare',
            'tags': ['healthcare', 'medical diagnosis', 'patient care', 'clinical ML'],
            'level': 'Advanced',
            'duration': '12 weeks',
            'is_free': False,
            'description': 'Apply ML to healthcare and medical applications',
            'curriculum': ['Medical Data Analysis', 'Diagnosis Models', 'Patient Care', 'Clinical Trials'],
            'prerequisites': ['ML Advanced', 'Healthcare Knowledge'],
            'skills_gained': ['Medical ML', 'Clinical Analysis', 'Healthcare Analytics'],
            'tools_used': ['Python', 'scikit-learn', 'MONAI'],
            'image': '🏥'
        },
        {
            'id': 'ML010',
            'title': 'ML for Cybersecurity',
            'category': 'Machine Learning',
            'subcategory': 'Security',
            'tags': ['cybersecurity', 'threat detection', 'security analytics', 'defense'],
            'level': 'Advanced',
            'duration': '10 weeks',
            'is_free': False,
            'description': 'Use ML techniques for cybersecurity applications',
            'curriculum': ['Security Analytics', 'Threat Detection', 'Network Security', 'Defense Strategies'],
            'prerequisites': ['ML Advanced', 'Security Basics'],
            'skills_gained': ['Security Analytics', 'Threat Detection', 'Defense Systems'],
            'tools_used': ['Python', 'Security Tools', 'TensorFlow'],
            'image': '🔒'
        },
        # Deep Learning Courses
{
    'id': 'DL001',
    'title': 'Deep Learning Fundamentals',
    'category': 'Deep Learning',
    'subcategory': 'Fundamentals',
    'tags': ['neural networks', 'deep learning', 'AI', 'backpropagation'],
    'level': 'Beginner',
    'duration': '8 weeks',
    'is_free': True,
    'description': 'Comprehensive introduction to deep learning concepts and neural networks',
    'curriculum': ['Neural Network Basics', 'Backpropagation', 'Optimization', 'Model Architecture'],
    'prerequisites': ['Python', 'Linear Algebra', 'Calculus'],
    'skills_gained': ['Neural Network Design', 'Model Training', 'Deep Learning Basics'],
    'tools_used': ['PyTorch', 'TensorFlow', 'Jupyter'],
    'image': '🧠'
},
{
    'id': 'DL002',
    'title': 'Computer Vision Deep Learning',
    'category': 'Deep Learning',
    'subcategory': 'Computer Vision',
    'tags': ['CNN', 'image processing', 'object detection', 'computer vision'],
    'level': 'Intermediate',
    'duration': '10 weeks',
    'is_free': False,
    'description': 'Master deep learning techniques for computer vision applications',
    'curriculum': ['CNN Architecture', 'Object Detection', 'Image Segmentation', 'Transfer Learning'],
    'prerequisites': ['Deep Learning Fundamentals', 'Python'],
    'skills_gained': ['Computer Vision', 'CNN Development', 'Image Processing'],
    'tools_used': ['PyTorch', 'OpenCV', 'TensorFlow'],
    'image': '👁️'
},
{
    'id': 'DL003',
    'title': 'GANs and Generative Models',
    'category': 'Deep Learning',
    'subcategory': 'Generative Models',
    'tags': ['GANs', 'VAE', 'generative models', 'deep learning'],
    'level': 'Advanced',
    'duration': '12 weeks',
    'is_free': False,
    'description': 'Learn to create and train generative adversarial networks',
    'curriculum': ['GAN Architecture', 'Training Strategies', 'VAE', 'Applications'],
    'prerequisites': ['Advanced Deep Learning', 'Computer Vision'],
    'skills_gained': ['GAN Development', 'Model Architecture', 'Generative Models'],
    'tools_used': ['PyTorch', 'TensorFlow', 'NVIDIA APIs'],
    'image': '🎨'
},
{
    'id': 'DL004',
    'title': 'Deep Reinforcement Learning',
    'category': 'Deep Learning',
    'subcategory': 'RL',
    'tags': ['DRL', 'policy networks', 'Q-learning', 'AI agents'],
    'level': 'Advanced',
    'duration': '10 weeks',
    'is_free': False,
    'description': 'Advanced course on deep reinforcement learning',
    'curriculum': ['DRL Fundamentals', 'Policy Networks', 'Deep Q-Learning', 'Actor-Critic Methods'],
    'prerequisites': ['Deep Learning', 'Reinforcement Learning'],
    'skills_gained': ['DRL Implementation', 'Policy Design', 'Agent Development'],
    'tools_used': ['PyTorch', 'OpenAI Gym', 'TensorFlow'],
    'image': '🎮'
},
{
    'id': 'DL005',
    'title': 'Advanced Neural Architectures',
    'category': 'Deep Learning',
    'subcategory': 'Architecture',
    'tags': ['neural architecture', 'model design', 'optimization'],
    'level': 'Advanced',
    'duration': '12 weeks',
    'is_free': False,
    'description': 'Study and implement advanced neural network architectures',
    'curriculum': ['Architecture Design', 'Performance Optimization', 'Modern Architectures', 'Research Papers'],
    'prerequisites': ['Advanced Deep Learning', 'Research Experience'],
    'skills_gained': ['Architecture Design', 'Model Optimization', 'Research Skills'],
    'tools_used': ['PyTorch', 'TensorFlow', 'Research Papers'],
    'image': '🏗️'
},
{
    'id': 'DL006',
    'title': 'Deep Learning for Audio',
    'category': 'Deep Learning',
    'subcategory': 'Audio',
    'tags': ['audio processing', 'speech recognition', 'music generation'],
    'level': 'Intermediate',
    'duration': '8 weeks',
    'is_free': False,
    'description': 'Apply deep learning to audio processing and generation',
    'curriculum': ['Audio Processing', 'Speech Recognition', 'Music Generation', 'Sound Classification'],
    'prerequisites': ['Deep Learning Fundamentals', 'Signal Processing'],
    'skills_gained': ['Audio Processing', 'Speech Recognition', 'Sound Generation'],
    'tools_used': ['PyTorch', 'Librosa', 'TensorFlow'],
    'image': '🎵'
},
{
    'id': 'DL007',
    'title': 'Transfer Learning in Deep Learning',
    'category': 'Deep Learning',
    'subcategory': 'Transfer Learning',
    'tags': ['transfer learning', 'model adaptation', 'fine-tuning'],
    'level': 'Intermediate',
    'duration': '6 weeks',
    'is_free': False,
    'description': 'Master transfer learning techniques for deep learning',
    'curriculum': ['Transfer Learning Basics', 'Fine-tuning Strategies', 'Domain Adaptation', 'Applications'],
    'prerequisites': ['Deep Learning Fundamentals'],
    'skills_gained': ['Transfer Learning', 'Model Adaptation', 'Fine-tuning'],
    'tools_used': ['PyTorch', 'TensorFlow', 'Hugging Face'],
    'image': '🔄'
},
{
    'id': 'DL008',
    'title': 'Deep Learning for Time Series',
    'category': 'Deep Learning',
    'subcategory': 'Time Series',
    'tags': ['time series', 'RNN', 'LSTM', 'forecasting'],
    'level': 'Intermediate',
    'duration': '8 weeks',
    'is_free': False,
    'description': 'Apply deep learning to time series analysis',
    'curriculum': ['RNN Architecture', 'LSTM Networks', 'Forecasting', 'Sequence Modeling'],
    'prerequisites': ['Deep Learning Fundamentals', 'Time Series Basics'],
    'skills_gained': ['Time Series Analysis', 'Sequence Modeling', 'Forecasting'],
    'tools_used': ['PyTorch', 'TensorFlow', 'Prophet'],
    'image': '📈'
},
{
    'id': 'DL009',
    'title': 'Deep Learning for Robotics',
    'category': 'Deep Learning',
    'subcategory': 'Robotics',
    'tags': ['robotics', 'control systems', 'perception', 'navigation'],
    'level': 'Advanced',
    'duration': '12 weeks',
    'is_free': False,
    'description': 'Implement deep learning in robotics applications',
    'curriculum': ['Robotic Perception', 'Control Systems', 'Navigation', 'Integration'],
    'prerequisites': ['Deep Learning', 'Robotics Basics'],
    'skills_gained': ['Robotic Systems', 'Perception', 'Control'],
    'tools_used': ['PyTorch', 'ROS', 'Simulation Tools'],
    'image': '🤖'
},
{
    'id': 'DL010',
    'title': 'Deep Learning for Medical Imaging',
    'category': 'Deep Learning',
    'subcategory': 'Healthcare',
    'tags': ['medical imaging', 'diagnosis', 'healthcare', 'CNN'],
    'level': 'Advanced',
    'duration': '10 weeks',
    'is_free': False,
    'description': 'Apply deep learning to medical image analysis',
    'curriculum': ['Medical Imaging', 'Diagnosis Models', 'Segmentation', 'Clinical Applications'],
    'prerequisites': ['Deep Learning', 'Healthcare Knowledge'],
    'skills_gained': ['Medical Imaging', 'Diagnosis', 'Healthcare AI'],
    'tools_used': ['PyTorch', 'MONAI', 'TensorFlow'],
    'image': '🏥'
},

# NLP Courses
{
    'id': 'NLP001',
    'title': 'Natural Language Processing Fundamentals',
    'category': 'NLP',
    'subcategory': 'Fundamentals',
    'tags': ['NLP', 'text processing', 'linguistics', 'language'],
    'level': 'Beginner',
    'duration': '8 weeks',
    'is_free': True,
    'description': 'Introduction to natural language processing concepts',
    'curriculum': ['Text Processing', 'Tokenization', 'Part of Speech Tagging', 'Named Entity Recognition'],
    'prerequisites': ['Python', 'Machine Learning Basics'],
    'skills_gained': ['Text Processing', 'NLP Basics', 'Language Analysis'],
    'tools_used': ['NLTK', 'spaCy', 'Python'],
    'image': '📝'
},
{
    'id': 'NLP002',
    'title': 'Advanced NLP with Transformers',
    'category': 'NLP',
    'subcategory': 'Transformers',
    'tags': ['transformers', 'BERT', 'attention', 'deep learning'],
    'level': 'Advanced',
    'duration': '12 weeks',
    'is_free': False,
    'description': 'Master transformer architectures for NLP',
    'curriculum': ['Transformer Architecture', 'Attention Mechanisms', 'BERT', 'GPT Models'],
    'prerequisites': ['NLP Fundamentals', 'Deep Learning'],
    'skills_gained': ['Transformer Models', 'Advanced NLP', 'Model Training'],
    'tools_used': ['PyTorch', 'Hugging Face', 'TensorFlow'],
    'image': '🤖'
},
{
    'id': 'NLP003',
    'title': 'Text Generation and Language Models',
    'category': 'NLP',
    'subcategory': 'Generation',
    'tags': ['text generation', 'language models', 'GPT', 'creative AI'],
    'level': 'Intermediate',
    'duration': '10 weeks',
    'is_free': False,
    'description': 'Learn to build and train language models',
    'curriculum': ['Language Modeling', 'Generation Techniques', 'Fine-tuning', 'Applications'],
    'prerequisites': ['NLP Fundamentals', 'Deep Learning'],
    'skills_gained': ['Text Generation', 'Language Modeling', 'Model Training'],
    'tools_used': ['PyTorch', 'Hugging Face', 'OpenAI API'],
    'image': '✍️'
},
{
    'id': 'NLP004',
    'title': 'Sentiment Analysis and Opinion Mining',
    'category': 'NLP',
    'subcategory': 'Sentiment Analysis',
    'tags': ['sentiment analysis', 'opinion mining', 'emotion detection'],
    'level': 'Intermediate',
    'duration': '6 weeks',
    'is_free': False,
    'description': 'Master sentiment analysis techniques',
    'curriculum': ['Sentiment Classification', 'Opinion Mining', 'Emotion Detection', 'Applications'],
    'prerequisites': ['NLP Fundamentals', 'Machine Learning'],
    'skills_gained': ['Sentiment Analysis', 'Text Classification', 'Emotion Detection'],
    'tools_used': ['NLTK', 'scikit-learn', 'Transformers'],
    'image': '😊'
},
{
    'id': 'NLP005',
    'title': 'Machine Translation Systems',
    'category': 'NLP',
    'subcategory': 'Translation',
    'tags': ['translation', 'sequence-to-sequence', 'multilingual'],
    'level': 'Advanced',
    'duration': '10 weeks',
    'is_free': False,
    'description': 'Build machine translation systems',
    'curriculum': ['Translation Models', 'Sequence-to-Sequence', 'Attention', 'Evaluation'],
    'prerequisites': ['Advanced NLP', 'Deep Learning'],
    'skills_gained': ['Machine Translation', 'Model Development', 'Evaluation'],
    'tools_used': ['PyTorch', 'TensorFlow', 'Transformers'],
    'image': '🌐'
},
{
    'id': 'NLP006',
    'title': 'Question Answering Systems',
    'category': 'NLP',
    'subcategory': 'QA Systems',
    'tags': ['question answering', 'information retrieval', 'BERT'],
    'level': 'Advanced',
    'duration': '8 weeks',
    'is_free': False,
    'description': 'Develop question answering systems',
    'curriculum': ['QA Architecture', 'Information Retrieval', 'Answer Generation', 'Evaluation'],
    'prerequisites': ['Advanced NLP', 'Transformers'],
    'skills_gained': ['QA Systems', 'Information Retrieval', 'System Design'],
    'tools_used': ['Hugging Face', 'PyTorch', 'Elasticsearch'],
    'image': '❓'
},
{
    'id': 'NLP007',
    'title': 'Chatbot Development',
    'category': 'NLP',
    'subcategory': 'Chatbots',
    'tags': ['chatbots', 'conversational AI', 'dialogue systems'],
    'level': 'Intermediate',
    'duration': '8 weeks',
    'is_free': False,
    'description': 'Build intelligent chatbot systems',
    'curriculum': ['Dialogue Systems', 'Intent Recognition', 'Response Generation', 'Integration'],
    'prerequisites': ['NLP Fundamentals', 'Python'],
    'skills_gained': ['Chatbot Development', 'Dialogue Systems', 'Integration'],
    'tools_used': ['Rasa', 'DialogFlow', 'Python'],
    'image': '💬'
},
{
    'id': 'NLP008',
    'title': 'Information Extraction',
    'category': 'NLP',
    'subcategory': 'Information Extraction',
    'tags': ['IE', 'named entity recognition', 'relation extraction'],
    'level': 'Intermediate',
    'duration': '6 weeks',
    'is_free': False,
    'description': 'Master information extraction techniques',
    'curriculum': ['Named Entity Recognition', 'Relation Extraction', 'Event Extraction', 'Applications'],
    'prerequisites': ['NLP Fundamentals'],
    'skills_gained': ['Information Extraction', 'NER', 'Relation Mining'],
    'tools_used': ['spaCy', 'Stanford NLP', 'GATE'],
    'image': '📑'
},
{
    'id': 'NLP009',
    'title': 'Text Summarization',
    'category': 'NLP',
    'subcategory': 'Summarization',
    'tags': ['summarization', 'text analysis', 'content generation'],
    'level': 'Advanced',
    'duration': '8 weeks',
    'is_free': False,
    'description': ' ',
},
      {
        'id': 'NLP001',
        'title': 'Text Classification Basics',
        'category': 'NLP',
        'subcategory': 'Text Classification',
        'tags': ['classification', 'sentiment analysis'],
        'level': 'Beginner',
        'duration': '4 weeks',
        'is_free': True,
        'description': 'Learn how to classify text data with machine learning.',
        'curriculum': ['Text Preprocessing', 'Feature Engineering', 'Model Building', 'Evaluation'],
        'prerequisites': ['Basic Python'],
        'skills_gained': ['Text Classification', 'Sentiment Analysis'],
        'tools_used': ['scikit-learn', 'NLTK'],
        'image': '📚'
    },
    {
        'id': 'NLP002',
        'title': 'Advanced Sentiment Analysis',
        'category': 'NLP',
        'subcategory': 'Sentiment Analysis',
        'tags': ['deep learning', 'sentiment'],
        'level': 'Advanced',
        'duration': '8 weeks',
        'is_free': False,
        'description': 'Dive deep into advanced techniques for sentiment analysis.',
        'curriculum': ['Deep Learning Models', 'Transformers', 'Fine-Tuning BERT'],
        'prerequisites': ['Deep Learning Fundamentals'],
        'skills_gained': ['BERT', 'Sentiment Analysis'],
        'tools_used': ['Hugging Face', 'TensorFlow'],
        'image': '🤖'
    },
    {
        'id': 'NLP003',
        'title': 'Text Summarization',
        'category': 'NLP',
        'subcategory': 'Text Summarization',
        'tags': ['abstractive', 'extractive'],
        'level': 'Intermediate',
        'duration': '6 weeks',
        'is_free': False,
        'description': 'Master text summarization techniques.',
        'curriculum': ['Extractive Summarization', 'Abstractive Summarization', 'Applications'],
        'prerequisites': ['NLP Fundamentals'],
        'skills_gained': ['Summarization', 'Text Processing'],
        'tools_used': ['spaCy', 'Transformers'],
        'image': '📝'
    },
    {
        'id': 'NLP004',
        'title': 'Language Modeling',
        'category': 'NLP',
        'subcategory': 'Language Modeling',
        'tags': ['language models', 'GPT'],
        'level': 'Advanced',
        'duration': '10 weeks',
        'is_free': False,
        'description': 'Learn to build and fine-tune language models.',
        'curriculum': ['N-grams', 'Recurrent Neural Networks', 'Transformers'],
        'prerequisites': ['Deep Learning'],
        'skills_gained': ['Language Modeling', 'GPT Fine-Tuning'],
        'tools_used': ['PyTorch', 'Transformers'],
        'image': '📜'
    },
    {
        'id': 'Projects001',
        'title': 'Chatbot Development',
        'category': 'Projects',
        'subcategory': 'NLP Projects',
        'tags': ['chatbots', 'dialog systems'],
        'level': 'Intermediate',
        'duration': '8 weeks',
        'is_free': False,
        'description': 'Develop intelligent chatbots for real-world applications.',
        'curriculum': ['Dialog Flow', 'Intent Recognition', 'Response Generation'],
        'prerequisites': ['NLP Fundamentals'],
        'skills_gained': ['Chatbot Development', 'Conversational AI'],
        'tools_used': ['Dialogflow', 'Rasa'],
        'image': '💬'
    },
    {
        'id': 'Projects002',
        'title': 'Fake News Detection',
        'category': 'Projects',
        'subcategory': 'Machine Learning Projects',
        'tags': ['text classification', 'fake news'],
        'level': 'Intermediate',
        'duration': '6 weeks',
        'is_free': True,
        'description': 'Build a system to detect fake news using NLP.',
        'curriculum': ['Data Collection', 'Text Vectorization', 'Model Training'],
        'prerequisites': ['Basic Machine Learning'],
        'skills_gained': ['Fake News Detection', 'Text Classification'],
        'tools_used': ['sklearn', 'NLTK'],
        'image': '📰'
    },
    {
        'id': 'Tools001',
        'title': 'Mastering spaCy',
        'category': 'Tools',
        'subcategory': 'NLP Tools',
        'tags': ['spaCy', 'NLP library'],
        'level': 'Beginner',
        'duration': '3 weeks',
        'is_free': True,
        'description': 'Learn the fundamentals of spaCy for NLP.',
        'curriculum': ['Tokenization', 'NER', 'POS Tagging'],
        'prerequisites': ['Basic Python'],
        'skills_gained': ['NLP with spaCy', 'NER'],
        'tools_used': ['spaCy'],
        'image': '🔧'
    },
    {
        'id': 'Tools002',
        'title': 'Hugging Face Transformers',
        'category': 'Tools',
        'subcategory': 'NLP Tools',
        'tags': ['Transformers', 'BERT'],
        'level': 'Intermediate',
        'duration': '4 weeks',
        'is_free': False,
        'description': 'Leverage Hugging Face Transformers for NLP tasks.',
        'curriculum': ['Installing Transformers', 'Pre-trained Models', 'Fine-tuning'],
        'prerequisites': ['NLP Fundamentals'],
        'skills_gained': ['Transformers', 'Fine-Tuning'],
        'tools_used': ['Hugging Face'],
        'image': '🚀'
    },
    {
        'id': 'Tools003',
        'title': 'Stanford NLP Toolkit',
        'category': 'Tools',
        'subcategory': 'NLP Tools',
        'tags': ['Stanford NLP', 'CoreNLP'],
        'level': 'Intermediate',
        'duration': '5 weeks',
        'is_free': False,
        'description': 'Master Stanford NLP tools for advanced tasks.',
        'curriculum': ['NER with CoreNLP', 'Dependency Parsing', 'Coreference Resolution'],
        'prerequisites': ['NLP Basics'],
        'skills_gained': ['CoreNLP Usage', 'Advanced NLP'],
        'tools_used': ['Stanford CoreNLP'],
        'image': '🔍'
    },
    {
        'id': 'Tools004',
        'title': 'Exploring GATE',
        'category': 'Tools',
        'subcategory': 'NLP Tools',
        'tags': ['GATE', 'text analysis'],
        'level': 'Advanced',
        'duration': '7 weeks',
        'is_free': False,
        'description': 'Learn to use GATE for large-scale text analysis.',
        'curriculum': ['Text Processing', 'Annotation', 'Custom Pipelines'],
        'prerequisites': ['NLP Fundamentals'],
        'skills_gained': ['GATE Usage', 'Text Annotation'],
        'tools_used': ['GATE'],
        'image': '⚙️'
    },

        # Add all other courses from your database here
    ]
}
class EnhancedCourseSearch:
    def __init__(self):
        self.model = load_model()
        self.courses = course_database['courses']
        self.course_embeddings = self._compute_course_embeddings()

    def _compute_course_embeddings(self) -> np.ndarray:
        """Compute embeddings for all courses"""
        course_texts = [
            f"{course['title']} {course['description']} {' '.join(course['tags'])}"
            for course in self.courses
        ]
        return self.model.encode(course_texts)

    def char_similarity(self, str1: str, str2: str) -> float:
        """Calculate character-based similarity between two strings"""
        return SequenceMatcher(None, str1.lower(), str2.lower()).ratio()

    def find_similar_by_char(self, query: str, threshold: float = 0.3) -> List[Dict]:
        """Find courses with similar character patterns in title or tags"""
        similar_courses = []
        for course in self.courses:
            # Check title similarity
            title_sim = self.char_similarity(query, course['title'])
            
            # Check tag similarities
            tag_sims = [self.char_similarity(query, tag) for tag in course['tags']]
            max_tag_sim = max(tag_sims) if tag_sims else 0
            
            # Take the maximum similarity score
            max_sim = max(title_sim, max_tag_sim)
            
            if max_sim > threshold:
                similar_courses.append({
                    "course": course,
                    "char_similarity": max_sim
                })
        
        # Sort by similarity score
        similar_courses.sort(key=lambda x: x['char_similarity'], reverse=True)
        return similar_courses

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Semantic search using transformer embeddings"""
        query_embedding = self.model.encode(query)
        similarities = cosine_similarity([query_embedding], self.course_embeddings)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        
        results = []
        for idx in top_indices:
            course = self.courses[idx]
            similarity_score = similarities[idx]
            results.append({
                "course": course,
                "similarity_score": float(similarity_score)
            })
        return results

    def combined_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """Combine semantic and character-based search results"""
        semantic_results = self.semantic_search(query, top_k)
        char_results = self.find_similar_by_char(query)
        
        # Combine and deduplicate results
        seen_titles = set()
        combined_results = []
        
        for result in semantic_results + char_results:
            course_title = result['course']['title']
            if course_title not in seen_titles:
                seen_titles.add(course_title)
                combined_results.append(result)
        
        return combined_results[:top_k]

    def filter_courses(self, category=None, level=None, duration=None, price=None):
        """Filter courses based on multiple criteria"""
        filtered = self.courses.copy()
        
        if category and category != "All Courses":
            filtered = [c for c in filtered if c['category'] == category]
        
        if level and level != "All Levels":
            filtered = [c for c in filtered if c['level'] == level]
            
        if duration and duration != "Any Duration":
            filtered = [c for c in filtered if duration in c['duration']]
            
        if price and price != "All":
            filtered = [c for c in filtered if 
                      (c['is_free'] and price == "Free") or 
                      (not c['is_free'] and price == "Paid")]
            
        return filtered

def display_course_card(course, similarity_score=None, char_similarity=None):
    """Enhanced course card display with both similarity scores"""
    tags_html = " ".join([f'<span class="badge badge-secondary">{tag}</span>' for tag in course['tags']])
    
    # Add both similarity scores if available
    similarity_html = ""
    if similarity_score is not None:
        similarity_html += f'<span class="badge badge-warning">Semantic Score: {similarity_score:.2f}</span>'
    if char_similarity is not None:
        similarity_html += f'<span class="badge badge-info">Character Score: {char_similarity:.2f}</span>'
    
    card_html = f"""
    <div style="border:1px solid #ddd; padding:15px; border-radius:10px; margin-bottom:15px;">
        <div style="display:flex; justify-content:space-between; align-items:start;">
            <h3>{course['image']} {course['title']}</h3>
            <span class="badge badge-{'success' if course['is_free'] else 'primary'}">
                {"Free" if course['is_free'] else "Premium"}
            </span>
        </div>
        <p>{course['description']}</p>
        <div style="margin-top:10px;">
            <span class="badge badge-info">{course['level']}</span>
            <span class="badge badge-info">{course['duration']}</span>
            {similarity_html}
        </div>
        <div style="margin-top:10px;">
            {tags_html}
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

def main():
    st.set_page_config(page_title="Enhanced Course Search", layout="wide")
    
    # Custom CSS (same as before)
    st.markdown("""
    <style>
    .badge {
        padding: 3px 8px;
        border-radius: 15px;
        font-size: 12px;
        margin-right: 5px;
    }
    .badge-secondary { background-color: #f0f0f0; }
    .badge-success { background-color: #28a745; color: white; }
    .badge-primary { background-color: #007bff; color: white; }
    .badge-info { background-color: #17a2b8; color: white; }
    .badge-warning { background-color: #ffc107; }
    </style>
    """, unsafe_allow_html=True)

    # Initialize search engine
    search_engine = EnhancedCourseSearch()

    # Rest of the UI setup (same as before)
    st.sidebar.title("📚 Course Categories")
    categories = ["All Courses", "Business Analytics", "Generative AI", "Machine Learning", 
                 "Deep Learning", "NLP", "Projects", "Tools"]
    selected_category = st.sidebar.selectbox("Select Category", categories)

    st.title("Analytics Vidhya Free Courses Search Platform")
    st.markdown("### 🔍 Find Your Perfect Course")

    # Search and filters
    col1, col2 = st.columns([3, 1])
    with col1:
        search_query = st.text_input("Search courses", 
                                   placeholder="Try 'machine learning for beginners' or 'advanced deep learning'")
    with col2:
        sort_by = st.selectbox("Sort by", ["Relevance", "Duration", "Level"])

    # Filters row
    st.markdown("### Quick Filters")
    filter_cols = st.columns(4)
    with filter_cols[0]:
        level = st.selectbox("Level", ["All Levels", "Beginner", "Intermediate", "Advanced"])
    with filter_cols[1]:
        duration = st.selectbox("Duration", ["Any Duration", "4 weeks", "6 weeks", "8 weeks", "10 weeks"])
    with filter_cols[2]:
        price = st.selectbox("Price", ["All", "Free", "Paid"])
    with filter_cols[3]:
        format_type = st.selectbox("Format", ["All Formats", "Course", "Project", "Workshop"])

    # Apply combined search and filters
    if search_query:
        results = search_engine.combined_search(search_query)
        filtered_results = [
            result for result in results
            if result['course'] in search_engine.filter_courses(
                category=selected_category, level=level, 
                duration=duration, price=price
            )
        ]
        
        st.markdown(f"### Search Results ({len(filtered_results)} courses)")
        if not filtered_results:
            st.info("No courses found matching your criteria. Try adjusting your filters or search terms.")
        else:
            for result in filtered_results:
                display_course_card(
                    result['course'],
                    similarity_score=result.get('similarity_score'),
                    char_similarity=result.get('char_similarity')
                )
    else:
        filtered_courses = search_engine.filter_courses(
            category=selected_category, level=level,
            duration=duration, price=price
        )
        
        st.markdown(f"### Featured Courses ({len(filtered_courses)} courses)")
        for course in filtered_courses[:6]:
            display_course_card(course)

    # Analytics section (same as before)
    st.markdown("### 📊 Course Analytics")
    col1, col2 = st.columns(2)
    
    with col1:
        category_counts = pd.DataFrame(
            [(c['category'], 1) for c in search_engine.courses],
            columns=['Category', 'Count']
        ).groupby('Category').sum()
        
        fig1 = px.pie(
            values=category_counts['Count'],
            names=category_counts.index,
            title='Course Distribution by Category',
            hole=0.4
        )
        st.plotly_chart(fig1)
    
    with col2:
        level_counts = pd.DataFrame(
            [(c['level'], 1) for c in search_engine.courses],
            columns=['Level', 'Count']
        ).groupby('Level').sum()
        
        fig2 = px.bar(
            x=level_counts.index,
            y=level_counts['Count'],
            title='Courses by Difficulty Level',
            labels={'x': 'Level', 'y': 'Number of Courses'}
        )
        st.plotly_chart(fig2)

if __name__ == "__main__":
    main()