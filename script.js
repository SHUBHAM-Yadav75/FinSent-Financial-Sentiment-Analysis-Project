// Mock model - replace this with actual API calls to your model
class FinancialSentimentModel {
    constructor() {
        this.emotions = [
            'Joy', 'Fear', 'Anger', 'Sadness', 'Surprise', 
            'Trust', 'Anticipation', 'Disgust'
        ];
    }
    
    async analyze(text) {
        // Simulate API delay
        await new Promise(resolve => setTimeout(resolve, 1500));
        
        // Mock analysis based on keywords (replace with actual model calls)
        const words = text.toLowerCase();
        let sentiment = { positive: 0.33, neutral: 0.33, negative: 0.34 };
        let emotions = {};
        
        // Simple keyword-based mock analysis
        if (words.includes('profit') || words.includes('growth') || words.includes('surge') || words.includes('exceed')) {
            sentiment = { positive: 0.78, neutral: 0.15, negative: 0.07 };
            emotions = {
                Joy: 0.85, Trust: 0.72, Anticipation: 0.68, Surprise: 0.45,
                Fear: 0.12, Anger: 0.08, Sadness: 0.15, Disgust: 0.05
            };
        } else if (words.includes('decline') || words.includes('drop') || words.includes('loss') || words.includes('concern')) {
            sentiment = { positive: 0.08, neutral: 0.22, negative: 0.70 };
            emotions = {
                Joy: 0.15, Trust: 0.25, Anticipation: 0.30, Surprise: 0.55,
                Fear: 0.82, Anger: 0.45, Sadness: 0.70, Disgust: 0.38
            };
        } else {
            sentiment = { positive: 0.35, neutral: 0.45, negative: 0.20 };
            emotions = {
                Joy: 0.40, Trust: 0.55, Anticipation: 0.48, Surprise: 0.32,
                Fear: 0.25, Anger: 0.18, Sadness: 0.22, Disgust: 0.15
            };
        }
        
        return { sentiment, emotions };
    }
}

// Initialize the model and get DOM elements
const model = new FinancialSentimentModel();
const textInput = document.getElementById('textInput');
const analyzeBtn = document.getElementById('analyzeBtn');
const resultsSection = document.getElementById('resultsSection');
const emotionBars = document.getElementById('emotionBars');

// Sample text buttons
document.querySelectorAll('.sample-btn').forEach(btn => {
    btn.addEventListener('click', () => {
        textInput.value = btn.dataset.text;
        textInput.focus();
    });
});

// Analyze button click handler
analyzeBtn.addEventListener('click', async () => {
    const text = textInput.value.trim();
    
    if (!text) {
        alert('Please enter some text to analyze!');
        return;
    }
    
    // Show loading state
    analyzeBtn.innerHTML = '<div class="loading"><div class="spinner"></div>Analyzing...</div>';
    analyzeBtn.disabled = true;
    resultsSection.classList.add('results-hidden');
    
    try {
        // Analyze text
        const results = await model.analyze(text);
        
        // Update sentiment scores
        document.getElementById('positiveScore').textContent = 
            Math.round(results.sentiment.positive * 100) + '%';
        document.getElementById('neutralScore').textContent = 
            Math.round(results.sentiment.neutral * 100) + '%';
        document.getElementById('negativeScore').textContent = 
            Math.round(results.sentiment.negative * 100) + '%';
        
        // Update emotion bars
        emotionBars.innerHTML = '';
        Object.entries(results.emotions)
            .sort(([,a], [,b]) => b - a)
            .forEach(([emotion, score]) => {
                const bar = createEmotionBar(emotion, score);
                emotionBars.appendChild(bar);
            });
        
        // Show results
        resultsSection.classList.remove('results-hidden');
        resultsSection.scrollIntoView({ behavior: 'smooth' });
        
    } catch (error) {
        alert('Error analyzing text. Please try again.');
        console.error(error);
    } finally {
        // Reset button
        analyzeBtn.innerHTML = '🔍 Analyze Sentiment';
        analyzeBtn.disabled = false;
    }
});

function createEmotionBar(emotion, score) {
    const container = document.createElement('div');
    container.className = 'emotion-bar';
    
    container.innerHTML = `
        <div class="emotion-header">
            <span class="emotion-name">${emotion}</span>
            <span class="emotion-value">${Math.round(score * 100)}%</span>
        </div>
        <div class="progress-bar">
            <div class="progress-fill" style="width: ${score * 100}%"></div>
        </div>
    `;
    
    return container;
}

// Allow Enter key to trigger analysis (with Ctrl/Cmd)
textInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
        e.preventDefault();
        analyzeBtn.click();
    }
});

// API Integration Template
// Replace the FinancialSentimentModel class with this for real API integration:

/*
class FinancialSentimentModel {
    constructor(apiEndpoint) {
        this.apiEndpoint = apiEndpoint || 'http://localhost:5000/analyze';
    }
    
    async analyze(text) {
        try {
            const response = await fetch(this.apiEndpoint, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: text })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const data = await response.json();
            
            // Assuming your API returns data in the format:
            // {
            //   sentiment: { positive: 0.7, neutral: 0.2, negative: 0.1 },
            //   emotions: { Joy: 0.8, Fear: 0.2, ... }
            // }
            
            return data;
            
        } catch (error) {
            console.error('API Error:', error);
            throw new Error('Failed to analyze text. Please check your connection and try again.');
        }
    }
}
*/