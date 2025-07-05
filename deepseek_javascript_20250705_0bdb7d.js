// Fallback configuration if fetch fails
const fallbackConfig = {
    "settings": {
        "defaultLanguage": "en",
        "defaultMood": "friendly",
        "supportedLanguages": [
            {"code": "en", "name": "English"},
            {"code": "hi", "name": "हिंदी (Hindi)"},
            {"code": "es", "name": "Español"},
            {"code": "fr", "name": "Français"}
        ]
    }
};

// Then in your initialization:
appConfig = fallbackConfig;