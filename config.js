// Configuration file for Financial Sentiment Analyzer

const CONFIG = {
    // API Configuration
    API: {
        BASE_URL: 'http://localhost:5000',  // Change to your API URL
        ENDPOINTS: {
            ANALYZE: '/analyze',
            BATCH_ANALYZE: '/batch-analyze'
        },
        TIMEOUT: 30000, // 30 seconds
        RETRY_ATTEMPTS: 3
    },
    
    // Model Configuration
    MODEL: {
        MAX_TEXT_LENGTH: 5000,
        MIN_TEXT_LENGTH: 10,
        SUPPORTED_LANGUAGES: ['en'],
        
        // Emotion categories - update these based on your model
        EMOTIONS: [
            'Joy', 'Fear', 'Anger', 'Sadness', 
            'Surprise', 'Trust', 'Anticipation', 'Disgust'
        ],
        
        // Sentiment categories
        SENTIMENT_CATEGORIES: ['positive', 'neutral', 'negative']
    },
    
    // UI Configuration
    UI: {
        ANIMATION_DURATION: 800,
        SCROLL_BEHAVIOR: 'smooth',
        SHOW_CONFIDENCE_THRESHOLD: 0.1, // Hide emotions below 10%
        
        // Color scheme for different sentiments
        COLORS: {
            POSITIVE: '#10b981',
            NEGATIVE: '#ef4444',
            NEUTRAL: '#6b7280',
            PRIMARY_GRADIENT: 'linear-gradient(45deg, #667eea, #764ba2)',
            EMOTION_GRADIENT: 'linear-gradient(45deg, #667eea, #764ba2)'
        }
    },
    
    // Sample texts for demo
    SAMPLE_TEXTS: [
        {
            label: "Positive Earnings",
            text: "The company reported record-breaking quarterly earnings, exceeding analyst expectations by 15%. Stock price surged in after-hours trading."
        },
        {
            label: "Market Decline", 
            text: "Market volatility continues as inflation concerns weigh on investor sentiment. Major indices declined for the third consecutive session."
        },
        {
            label: "Neutral Fed News",
            text: "The Federal Reserve announced no changes to interest rates, maintaining current monetary policy stance."
        },
        {
            label: "Crypto Volatility",
            text: "Cryptocurrency markets experienced severe turbulence following regulatory concerns, with Bitcoin dropping 12% overnight."
        },
        {
            label: "IPO Launch",
            text: "The highly anticipated IPO launched successfully, with shares opening 40% above the initial pricing amid strong investor demand."
        },
        {
            label: "Merger News",
            text: "The proposed merger between the two tech giants faces regulatory scrutiny, with antitrust officials expressing concerns about market concentration."
        }
    ],
    
    // Error messages
    ERRORS: {
        EMPTY_TEXT: 'Please enter some text to analyze!',
        TEXT_TOO_SHORT: `Text must be at least ${this.MODEL?.MIN_TEXT_LENGTH || 10} characters long.`,
        TEXT_TOO_LONG: `Text cannot exceed ${this.MODEL?.MAX_TEXT_LENGTH || 5000} characters.`,
        API_ERROR: 'Error analyzing text. Please try again.',
        NETWORK_ERROR: 'Network error. Please check your connection and try again.',
        TIMEOUT_ERROR: 'Request timed out. Please try again.',
        GENERIC_ERROR: 'An unexpected error occurred. Please try again.'
    },
    
    // Feature flags
    FEATURES: {
        ENABLE_BATCH_ANALYSIS: false,
        ENABLE_EXPORT: true,
        ENABLE_HISTORY: false,
        ENABLE_REAL_TIME: false,
        ENABLE_DARK_MODE: false
    }
};

// Export for use in other files
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CONFIG;
} else if (typeof window !== 'undefined') {
    window.CONFIG = CONFIG;
}