// Global variables
let currentMood = 'friendly';
let currentLanguage = 'en';
let isListening = false;
let recognition;
let vantaEffect;
let appConfig = {}; // This will hold our configuration

// Initialize the application
document.addEventListener('DOMContentLoaded', async function() {
    try {
        // Load configuration first
        const response = await fetch('config.json');
        if (!response.ok) {
            throw new Error('Failed to load configuration');
        }
        appConfig = await response.json();
        
        // Set default language and mood from config
        currentLanguage = appConfig.settings.defaultLanguage;
        currentMood = appConfig.settings.defaultMood;
        
        // Update UI to reflect default language
        document.querySelector('#languageBtn span').textContent = 
            appConfig.settings.supportedLanguages.find(lang => lang.code === currentLanguage).name;
        
        // Initialize Vanta.js background
        vantaEffect = initVantaBackground();
        
        // Set up event listeners
        setupEventListeners();
        
        // Add welcome message
        addAIMessage(getWelcomeMessage());
        
    } catch (error) {
        console.error('Error initializing app:', error);
        // Fallback welcome message if config fails to load
        addAIMessage("Welcome to Ciyaar AI Trading Assistant! How can I help you today?");
    }
});

function getWelcomeMessage() {
    if (currentLanguage === 'hi') {
        return "नमस्ते! मैं Ciyaar AI हूँ, आपका ट्रेडिंग सहायक। आज मैं आपकी कैसे मदद कर सकता हूँ?";
    } else if (currentLanguage === 'es') {
        return "¡Hola! Soy Ciyaar AI, tu asistente de trading. ¿Cómo puedo ayudarte hoy?";
    } else if (currentLanguage === 'fr') {
        return "Bonjour ! Je suis Ciyaar AI, votre assistant de trading. Comment puis-je vous aider aujourd'hui ?";
    } else {
        return "Hello! I'm Ciyaar AI, your trading assistant. How can I help you today?";
    }
}

function setupEventListeners() {
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    document.getElementById('darkModeToggle').addEventListener('click', function() {
        document.body.classList.toggle('dark');
        if (document.body.classList.contains('dark')) {
            document.documentElement.style.backgroundColor = '#111827';
            vantaEffect.setOptions({ backgroundColor: 0x111827 });
        } else {
            document.documentElement.style.backgroundColor = '';
            vantaEffect.setOptions({ backgroundColor: 0xf1f5f9 });
        }
    });
    
    // Add other event listeners here...
}

// Rest of your existing functions...