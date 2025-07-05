// Initialize Vanta.js background
function initVantaBackground() {
    return VANTA.NET({
        el: "#aiCanvas",
        mouseControls: true,
        touchControls: true,
        gyroControls: false,
        minHeight: 200.00,
        minWidth: 200.00,
        scale: 1.00,
        scaleMobile: 1.00,
        color: 0x3b82f6,
        backgroundColor: 0xf1f5f9,
        points: 12.00,
        maxDistance: 22.00,
        spacing: 18.00
    });
}

// Global variables
let currentMood = 'friendly';
let currentLanguage = 'en';
let isListening = false;
let recognition;
let vantaEffect;

// DOM elements
const authModal = document.getElementById('authModal');
const voiceModal = document.getElementById('voiceModal');
const voiceStatus = document.getElementById('voiceStatus');
const voiceTranscript = document.getElementById('voiceTranscript');
const voiceAnimation = document.getElementById('voiceAnimation');
const startListeningBtn = document.getElementById('startListeningBtn');
const chatContainer = document.getElementById('chatContainer');
const userInput = document.getElementById('userInput');
const chartContainer = document.getElementById('chartContainer');

// Modal functions
function openAuthModal() {
    authModal.classList.remove('hidden');
}

function closeAuthModal() {
    authModal.classList.add('hidden');
}

function showLoginForm() {
    document.getElementById('signupForm').classList.add('hidden');
    document.getElementById('loginForm').classList.remove('hidden');
}

function showSignupForm() {
    document.getElementById('loginForm').classList.add('hidden');
    document.getElementById('signupForm').classList.remove('hidden');
}

function openVoiceModal() {
    voiceModal.classList.remove('hidden');
}

function closeVoiceModal() {
    voiceModal.classList.add('hidden');
    if (isListening) {
        stopVoiceRecognition();
    }
}

// Language functions
function changeLanguage(lang) {
    currentLanguage = lang;
    document.querySelector('#languageBtn span').textContent = 
        lang === 'en' ? 'English' : 
        lang === 'hi' ? 'हिंदी' : 
        lang === 'es' ? 'Español' : 'Français';
        
    updateChatForLanguage(lang);
}

function updateChatForLanguage(lang) {
    if (lang === 'hi') {
        addAIMessage("भाषा हिंदी में बदल दी गई है! मैं आपकी कैसे मदद कर सकता हूँ?", true);
    } else if (lang === 'es') {
        addAIMessage("¡Idioma cambiado al español! ¿Cómo puedo ayudarte?", true);
    } else if (lang === 'fr') {
        addAIMessage("Langue changée en français! Comment puis-je vous aider?", true);
    } else {
        addAIMessage("Language changed to English! How can I help you?", true);
    }
}

// Mood functions
function setMood(mood) {
    currentMood = mood;
    document.querySelectorAll('.mood-selector button').forEach(btn => {
        btn.classList.remove('active');
    });
    event.target.classList.add('active');
    
    if (mood === 'hindi') {
        addAIMessage("मूड हिंदी मोड में बदल दिया गया है! अब मैं आपसे दोस्त की तरह बात करूंगा।", true);
    } else {
        addAIMessage(`My mood is now set to ${mood}. How can I assist you?`, true);
    }
}

// Voice recognition
function startVoiceRecognition() {
    if (!('webkitSpeechRecognition' in window)) {
        voiceStatus.textContent = "Voice recognition not supported in your browser";
        return;
    }
    
    recognition = new webkitSpeechRecognition();
    recognition.continuous = true;
    recognition.interimResults = true;
    recognition.lang = document.getElementById('voiceLanguage').value;
    
    recognition.onstart = function() {
        isListening = true;
        startListeningBtn.innerHTML = '<i class="fas fa-microphone-slash mr-2"></i><span>Stop Listening</span>';
        voiceStatus.textContent = "Listening... Speak now";
        voiceAnimation.innerHTML = '<div class="voice-wave"><span></span><span></span><span></span><span></span><span></span></div>';
    };
    
    recognition.onresult = function(event) {
        let interimTranscript = '';
        let finalTranscript = '';
        
        for (let i = event.resultIndex; i < event.results.length; i++) {
            const transcript = event.results[i][0].transcript;
            if (event.results[i].isFinal) {
                finalTranscript += transcript;
            } else {
                interimTranscript += transcript;
            }
        }
        
        voiceTranscript.innerHTML = finalTranscript + '<span class="text-gray-400">' + interimTranscript + '</span>';
        
        if (finalTranscript) {
            processVoiceCommand(finalTranscript);
        }
    };
    
    recognition.onerror = function(event) {
        console.error('Voice recognition error', event.error);
        voiceStatus.textContent = "Error: " + event.error;
        stopVoiceRecognition();
    };
    
    recognition.onend = function() {
        if (isListening) {
            recognition.start();
        }
    };
    
    recognition.start();
}

function stopVoiceRecognition() {
    if (recognition) {
        recognition.stop();
    }
    isListening = false;
    startListeningBtn.innerHTML = '<i class="fas fa-microphone mr-2"></i><span>Start Speaking</span>';
    voiceStatus.textContent = "Click the microphone to start speaking";
    voiceAnimation.innerHTML = '<div class="voice-wave"><span></span><span></span><span></span><span></span><span></span></div>';
}

function processVoiceCommand(command) {
    addUserMessage(command);
    command = command.toLowerCase();
    
    if (command.includes("reliance") || command.includes("रिलायंस")) {
        if (currentLanguage === 'hi') {
            addAIMessage("रिलायंस इंडस्ट्रीज का वर्तमान भाव ₹2,845.60 है, जो आज 1.8% की वृद्धि के साथ कारोबार कर रहा है। शेयर ने अपने 50-दिन और 200-दिन के मूविंग एवरेज को पार कर लिया है। आरएसआई 62 पर है जो हल्के ओवरबॉट स्थिति को दर्शाता है।", true);
        } else {
            addAIMessage("Reliance Industries is currently trading at ₹2,845.60, up 1.8% today. The stock has crossed above its 50-day and 200-day moving averages. RSI is at 62 indicating mildly overbought conditions.", true);
        }
    } else if (command.includes("nifty") || command.includes("निफ्टी")) {
        if (currentLanguage === 'hi') {
            addAIMessage("निफ्टी 50 वर्तमान में 22,415.65 पर कारोबार कर रहा है, जो आज 1.25% की वृद्धि के साथ है। प्रमुख समर्थन 22,350 और प्रतिरोध 22,500 स्तर पर है।", true);
        } else {
            addAIMessage("Nifty 50 is currently trading at 22,415.65, up 1.25% today. Key support is at 22,350 and resistance at 22,500 levels.", true);
        }
        showChart('NIFTY');
    } else if (command.includes("chart") || command.includes("चार्ट")) {
        if (command.includes("rsi") || command.includes("आरएसआई")) {
            addAIMessage("Showing chart with RSI indicator...", true);
            showChart(command.includes("nifty") ? 'NIFTY' : 'RELIANCE', 'rsi');
        } else if (command.includes("macd")) {
            addAIMessage("Showing chart with MACD indicator...", true);
            showChart(command.includes("nifty") ? 'NIFTY' : 'RELIANCE', 'macd');
        } else {
            addAIMessage("Showing chart...", true);
            showChart(command.includes("nifty") ? 'NIFTY' : 'RELIANCE');
        }
    } else if (command.includes("strategy") || command.includes("रणनीति")) {
        analyzeStrategy('breakout');
    } else {
        if (currentLanguage === 'hi') {
            addAIMessage("मैं आपके प्रश्न को समझ नहीं पाया। क्या आप कृपया इसे दोबारा कह सकते हैं?", true);
        } else {
            addAIMessage("I didn't understand your question. Could you please rephrase it?", true);
        }
    }
}

// Chat functions
function sendMessage() {
    const message = userInput.value.trim();
    if (message) {
        addUserMessage(message);
        userInput.value = '';
        
        setTimeout(() => {
            generateAIResponse(message);
        }, 500);
    }
}

function addUserMessage(message) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'flex justify-end';
    messageDiv.innerHTML = `
        <div class="bg-blue-100 text-gray-800 p-3 rounded-lg user-bubble max-w-xs lg:max-w-md dark:bg-blue-800 dark:bg-opacity-30 dark:text-white">
            <p>${message}</p>
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function addAIMessage(message, isVoice = false) {
    const messageDiv = document.createElement('div');
    messageDiv.className = 'flex';
    messageDiv.innerHTML = `
        <div class="w-8 h-8 bg-gradient-to-r from-blue-500 to-purple-600 rounded-full flex items-center justify-center text-white font-bold text-sm mr-2">
            CA
        </div>
        <div class="bg-blue-50 text-gray-800 p-3 rounded-lg chat-bubble max-w-xs lg:max-w-md dark:bg-blue-900 dark:bg-opacity-30 dark:text-white">
            <p>${message}</p>
            ${isVoice ? '<p class="text-xs text-gray-500 mt-1 dark:text-gray-400"><i class="fas fa-microphone"></i> Voice input processed</p>' : ''}
        </div>
    `;
    chatContainer.appendChild(messageDiv);
    chatContainer.scrollTop = chatContainer.scrollHeight;
}

function generateAIResponse(userMessage) {
    userMessage = userMessage.toLowerCase();
    
    if (userMessage.includes("hi") || userMessage.includes("hello") || userMessage.includes("namaste")) {
        if (currentLanguage === 'hi') {
            addAIMessage("नमस्ते! आज मैं आपकी कैसे मदद कर सकता हूँ?");
        } else {
            addAIMessage("Hello there! How can I help you today?");
        }
    } else if (userMessage.includes("reliance") || userMessage.includes("रिलायंस")) {
        if (currentLanguage === 'hi') {
            addAIMessage("रिलायंस इंडस्ट्रीज (RELIANCE.NS) - वर्तमान विश्लेषण:<br><br>वर्तमान मूल्य: ₹2,845.60<br>आज का परिवर्तन: +1.8% (₹50.20)<br>52 सप्ताह रेंज: ₹2,180 - ₹3,025<br><br>तकनीकी दृष्टिकोण: स्टॉक अपने 50-दिन और 200-दिन के मूविंग एवरेज से ऊपर कारोबार कर रहा है। आरएसआई 62 पर है जो हल्के ओवरबॉट स्थिति को दर्शाता है लेकिन तेजी का संकेत देता है।<br><br>मुख्य स्तर: समर्थन ₹2,780, प्रतिरोध ₹2,920");
        } else {
            addAIMessage("Reliance Industries (RELIANCE.NS) - Current Analysis:<br><br>Current Price: ₹2,845.60<br>Today's Change: +1.8% (₹50.20)<br>52 Week Range: ₹2,180 - ₹3,025<br><br>Technical View: Stock is trading above its 50-day and 200-day moving averages. RSI at 62 suggests mildly overbought conditions but with bullish momentum.<br><br>Key Levels: Support at ₹2,780, Resistance at ₹2,920");
        }
    } else if (userMessage.includes("nifty") || userMessage.includes("निफ्टी")) {
        if (currentLanguage === 'hi') {
            addAIMessage("निफ्टी 50 वर्तमान में 22,415.65 पर कारोबार कर रहा है, जो आज 1.25% की वृद्धि के साथ है।<br><br>तकनीकी दृष्टिकोण: निफ्टी ने हाल ही में 22,300 के प्रतिरोध स्तर को पार किया है और अब अगले प्रतिरोध 22,500 की ओर बढ़ रहा है। आरएसआई 58 पर है जो तटस्थ से तेजी का संकेत देता है।<br><br>मुख्य स्तर:<br>समर्थन: 22,350 और 22,200<br>प्रतिरोध: 22,500 और 22,650");
        } else {
            addAIMessage("Nifty 50 is currently trading at 22,415.65, up 1.25% today.<br><br>Technical View: Nifty has recently broken above the 22,300 resistance level and is now moving towards the next resistance at 22,500. RSI is at 58 indicating neutral to bullish momentum.<br><br>Key Levels:<br>Support: 22,350 and 22,200<br>Resistance: 22,500 and 22,650");
        }
        showChart('NIFTY');
    } else if (userMessage.includes("chart") || userMessage.includes("चार्ट")) {
        if (userMessage.includes("rsi") || userMessage.includes("आरएसआई")) {
            addAIMessage("Showing chart with RSI indicator...");
            showChart(userMessage.includes("nifty") ? 'NIFTY' : 'RELIANCE', 'rsi');
        } else if (userMessage.includes("macd")) {
            addAIMessage("Showing chart with MACD indicator...");
            showChart(userMessage.includes("nifty") ? 'NIFTY' : 'RELIANCE', 'macd');
        } else {
            addAIMessage("Showing chart...");
            showChart(userMessage.includes("nifty") ? 'NIFTY' : 'RELIANCE');
        }
    } else if (userMessage.includes("strategy") || userMessage.includes("रणनीति")) {
        analyzeStrategy('breakout');
    } else {
        if (currentLanguage === 'hi') {
            addAIMessage("क्षमा करें, मैं आपके प्रश्न को पूरी तरह से नहीं समझ पाया। क्या आप कृपया इसे अलग तरीके से पूछ सकते हैं? या फिर आप मुझसे स्टॉक, बाजार के रुझान, तकनीकी विश्लेषण, या ट्रेडिंग रणनीतियों के बारे में पूछ सकते हैं।");
        } else {
            addAIMessage("I'm sorry, I didn't fully understand your question. Could you please rephrase it? Or you can ask me about stocks, market trends, technical analysis, or trading strategies.");
        }
    }
}

// Chart functions
function showChart(symbol, indicator = null) {
    chartContainer.classList.remove('hidden');
    const ctx = document.getElementById('stockChart').getContext('2d');
    
    const labels = [];
    const data = [];
    const rsiData = [];
    const macdData = [];
    
    for (let i = 30; i >= 0; i--) {
        const date = new Date();
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString());
        
        const basePrice = symbol === 'NIFTY' ? 22000 : 2800;
        const randFactor = Math.random() * 2 - 1;
        const price = basePrice + Math.sin(i/3) * 200 + randFactor * 100;
        data.push(price);
        
        rsiData.push(30 + Math.sin(i/5) * 40 + randFactor * 10);
        macdData.push(-5 + Math.sin(i/4) * 10 + randFactor * 3);
    }
    
    if (window.stockChart) {
        window.stockChart.destroy();
    }
    
    const datasets = [
        {
            label: symbol + ' Price',
            data: data,
            borderColor: '#3b82f6',
            backgroundColor: 'rgba(59, 130, 246, 0.1)',
            tension: 0.1,
            yAxisID: 'y'
        }
    ];
    
    if (indicator === 'rsi') {
        datasets.push({
            label: 'RSI (14)',
            data: rsiData,
            borderColor: '#8b5cf6',
            backgroundColor: 'rgba(139, 92, 246, 0.1)',
            tension: 0.1,
            yAxisID: 'y1'
        });
    } else if (indicator === 'macd') {
        datasets.push({
            label: 'MACD',
            data: macdData,
            borderColor: '#ec4899',
            backgroundColor: 'rgba(236, 72, 153, 0.1)',
            tension: 0.1,
            yAxisID: 'y1'
        });
    }
    
    window.stockChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: labels,
            datasets: datasets
        },
        options: {
            responsive: true,
            interaction: {
                mode: 'index',
                intersect: false,
            },
            scales: {
                y: {
                    type: 'linear',
                    display: true,
                    position: 'left',
                },
                y1: {
                    type: 'linear',
                    display: indicator !== null,
                    position: 'right',
                    grid: {
                        drawOnChartArea: false,
                    },
                },
            }
        }
    });
    
    chartContainer.scrollIntoView({ behavior: 'smooth' });
}

// Trading functions
function addIndicator(indicator) {
    let message = "";
    if (currentLanguage === 'hi') {
        message = {
            'rsi': "RSI (सापेक्ष शक्ति सूचकांक) 14-अवधि का उपयोग ओवरबॉट और ओवरसोल्ड स्थितियों की पहचान के लिए किया जाता है। 70 से ऊपर ओवरबॉट और 30 से नीचे ओवरसोल्ड माना जाता है।",
            'macd': "MACD (मूविंग एवरेज कन्वर्जेंस डाइवर्जेंस) एक ट्रेंड-फॉलोइंग मोमेंटम इंडिकेटर है जो दो मूविंग एवरेज के बीच संबंध दिखाता है।",
            'bollinger': "बोलिंगर बैंड वोलेटिलिटी दिखाते हैं और कीमत के चरम स्तरों की पहचान करने में मदद करते हैं।",
            'sma': "सिंपल मूविंग एवरेज (SMA) एक निर्दिष्ट अवधि में सुरक्षा की औसत कीमत की गणना करता है।",
            'ema': "एक्सपोनेंशियल मूविंग एवरेज (EMA) SMA के समान है लेकिन हाल के डेटा को अधिक वजन देता है।",
            'stochastic': "स्टोकेस्टिक ऑसिलेटर वर्तमान कीमत की तुलना एक निर्दिष्ट अवधि में कीमत की रेंज से करता है।"
        }[indicator];
    } else {
        message = {
            'rsi': "RSI (Relative Strength Index) 14-period is used to identify overbought and oversold conditions. Above 70 is considered overbought, below 30 is oversold.",
            'macd': "MACD (Moving Average Convergence Divergence) is a trend-following momentum indicator that shows the relationship between two moving averages.",
            'bollinger': "Bollinger Bands show volatility and help identify price extremes.",
            'sma': "Simple Moving Average (SMA) calculates the average price of a security over a specified period.",
            'ema': "Exponential Moving Average (EMA) is similar to SMA but gives more weight to recent data.",
            'stochastic': "Stochastic Oscillator compares the current price to its price range over a specified period."
        }[indicator];
    }
    
    addAIMessage(message);
    
    if (chartContainer.classList.contains('hidden')) {
        showChart('NIFTY', indicator);
    } else {
        const currentSymbol = window.stockChart.data.datasets[0].label.split(' ')[0];
        showChart(currentSymbol, indicator);
    }
}

function analyzeStrategy(strategy) {
    let message = "";
    if (currentLanguage === 'hi') {
        message = {
            'breakout': "<b>ब्रेकआउट ट्रेडिंग रणनीति:</b><br><br>ब्रेकआउट ट्रेडिंग में प्रमुख समर्थन या प्रतिरोध स्तरों के ऊपर या नीचे कीमत के ब्रेकआउट की पहचान करना और उस दिशा में ट्रेड करना शामिल है।<br><br><b>वर्तमान बाजार में अनुप्रयोग:</b><br>निफ्टी ने हाल ही में 22,300 के प्रतिरोध स्तर को तोड़ दिया है। ब्रेकआउट की पुष्टि के लिए वॉल्यूम में वृद्धि देखी गई। अगला प्रतिरोध 22,500 पर है।<br><br><b>सुझाए गए ट्रेड:</b><br>- NIFTY में लॉन्ग पोजीशन लें, स्टॉप लॉस 22,250 के नीचे<br>- ब्रेकआउट स्टॉक जैसे TCS और INFY पर विचार करें",
            'pullback': "<b>पुलबैक ट्रेडिंग रणनीति:</b><br><br>पुलबैक रणनीति में एक मजबूत ट्रेंड में रिट्रेसमेंट के दौरान ट्रेड करना शामिल है, जिससे बेहतर एंट्री पॉइंट मिलता है।<br><br><b>वर्तमान बाजार में अनुप्रयोग:</b><br>IT सेक्टर में मजबूत अपट्रेंड है। TCS ₹3,950 के प्रतिरोध के पास पहुंचने के बाद छोटे पुलबैक की संभावना है।<br><br><b>सुझाए गए ट्रेड:</b><br>- TCS में ₹3,850 के आसपास पुलबैक पर खरीदारी करें, स्टॉप लॉस ₹3,800 के नीचे<br>- टार्गेट ₹4,100 (पिछले उच्च)",
            'mean-reversion': "<b>मीन रिवर्सन रणनीति:</b><br><br>मीन रिवर्सन इस धारणा पर आधारित है कि कीमतें समय के साथ उनके औसत स्तर पर वापस लौटती हैं।<br><br><b>वर्तमान बाजार में अनुप्रयोग:</b><br>HDFC बैंक अपने 200-दिन के मूविंग एवरेज (₹1,550) से नीचे आ गया है और अब ओवरसोल्ड RSI (28) दिखा रहा है।<br><br><b>सुझाए गए ट्रेड:</b><br>- HDFC बैंक में ₹1,530 के आसपास खरीदारी करें, स्टॉप लॉस ₹1,500 के नीचे<br>- टार्गेट ₹1,600 (200-DMA)"
        }[strategy];
    } else {
        message = {
            'breakout': "<b>Breakout Trading Strategy:</b><br><br>Breakout trading involves identifying breakouts above key resistance or below support levels and trading in the direction of the breakout.<br><br><b>Application in Current Market:</b><br>Nifty has recently broken above the 22,300 resistance level. Volume has increased confirming the breakout. Next resistance is at 22,500.<br><br><b>Suggested Trades:</b><br>- Go long on NIFTY, stop loss below 22,250<br>- Consider breakout stocks like TCS and INFY",
            'pullback': "<b>Pullback Trading Strategy:</b><br><br>The pullback strategy involves trading during retracements in a strong trend, providing better entry points.<br><br><b>Application in Current Market:</b><br>IT sector is in strong uptrend. TCS likely to see small pullback after approaching ₹3,950 resistance.<br><br><b>Suggested Trades:</b><br>- Buy TCS on pullback around ₹3,850, stop loss below ₹3,800<br>- Target ₹4,100 (previous high)",
            'mean-reversion': "<b>Mean Reversion Strategy:</b><br><br>Mean reversion is based on the idea that prices tend to return to their average levels over time.<br><br><b>Application in Current Market:</b><br>HDFC Bank has fallen below its 200-day moving average (₹1,550) and now shows oversold RSI (28).<br><br><b>Suggested Trades:</b><br>- Buy HDFC Bank around ₹1,530, stop loss below ₹1,500<br>- Target ₹1,600 (200-DMA)"
        }[strategy];
    }
    
    addAIMessage(message);
}

function showStockDetail(symbol) {
    const stocks = {
        'RELIANCE': {
            name: "Reliance Industries",
            price: "₹2,845.60",
            change: "+1.8%",
            high: "₹2,865.20",
            low: "₹2,810.40",
            pe: "28.5",
            sector: "Oil & Gas"
        },
        'TCS': {
            name: "Tata Consultancy Services",
            price: "₹3,912.45",
            change: "+2.3%",
            high: "₹3,925.00",
            low: "₹3,845.60",
            pe: "32.1",
            sector: "Information Technology"
        },
        'HDFCBANK': {
            name: "HDFC Bank",
            price: "₹1,542.80",
            change: "-0.7%",
            high: "₹1,560.40",
            low: "₹1,535.20",
            pe: "18.2",
            sector: "Banking"
        },
        'INFY': {
            name: "Infosys",
            price: "₹1,487.35",
            change: "+1.5%",
            high: "₹1,495.80",
            low: "₹1,465.40",
            pe: "26.8",
            sector: "Information Technology"
        },
        'BHARTIARTL': {
            name: "Bharti Airtel",
            price: "₹1,156.90",
            change: "+0.9%",
            high: "₹1,165.40",
            low: "₹1,145.60",
            pe: "58.3",
            sector: "Telecommunication"
        }
    };
    
    const stock = stocks[symbol];
    let message = "";
    
    if (currentLanguage === 'hi') {
        message = `
            <b>${stock.name} (${symbol})</b><br><br>
            वर्तमान मूल्य: ${stock.price}<br>
            आज का परिवर्तन: <span class="${stock.change.includes('+') ? 'text-green-500' : 'text-red-500'}">${stock.change}</span><br>
            दिन का उच्च: ${stock.high}<br>
            दिन का निम्न: ${stock.low}<br>
            P/E अनुपात: ${stock.pe}<br>
            सेक्टर: ${stock.sector}<br><br>
            <button onclick="showChart('${symbol}')" class="text-sm text-blue-600 hover:underline dark:text-blue-400">चार्ट देखें →</button>
        `;
    } else {
        message = `
            <b>${stock.name} (${symbol})</b><br><br>
            Current Price: ${stock.price}<br>
            Today's Change: <span class="${stock.change.includes('+') ? 'text-green-500' : 'text-red-500'}">${stock.change}</span><br>
            Day's High: ${stock.high}<br>
            Day's Low: ${stock.low}<br>
            P/E Ratio: ${stock.pe}<br>
            Sector: ${stock.sector}<br><br>
            <button onclick="showChart('${symbol}')" class="text-sm text-blue-600 hover:underline dark:text-blue-400">View Chart →</button>
        `;
    }
    
    addAIMessage(message);
}

function showNewsDetail(newsId) {
    const newsItems = {
        'n1': {
            headline: "RBI Keeps Repo Rate Unchanged at 6.5%",
            summary: "The Reserve Bank of India's Monetary Policy Committee (MPC) has decided to keep the repo rate unchanged at 6.5% in its bi-monthly policy review, in line with market expectations. The central bank maintained its stance of 'withdrawal of accommodation' to ensure inflation progressively aligns with the target while supporting growth.",
            impact: "Banking stocks showed mixed reactions, while rate-sensitive sectors like real estate remained stable."
        },
        'n2': {
            headline: "Infosys Q4 Results Beat Estimates",
            summary: "IT major Infosys reported a 12% year-on-year growth in net profit at ₹6,128 crore for the quarter ended March 2024, beating analyst estimates. The company announced a final dividend of ₹16 per share and gave a revenue growth guidance of 4-7% for FY25 in constant currency terms.",
            impact: "Infosys shares rose 2.5% in early trade, lifting other IT stocks as well."
        },
        'n3': {
            headline: "Global Markets Rally on Fed Rate Cut Hopes",
            summary: "Global stock markets rallied to record highs after US inflation data came in softer than expected, fueling hopes that the Federal Reserve may cut interest rates sooner than anticipated. The S&P 500 crossed 5,200 for the first time, while European and Asian markets also saw strong gains.",
            impact: "Indian markets expected to open higher tracking global cues, with IT stocks likely to benefit most."
        }
    };
    
    const news = newsItems[newsId];
    let message = "";
    
    if (currentLanguage === 'hi') {
        message = `
            <b>${news.headline}</b><br><br>
            ${news.summary}<br><br>
            <b>बाजार पर प्रभाव:</b> ${news.impact}<br><br>
            <button onclick="analyzeNewsImpact('${newsId}')" class="text-sm text-blue-600 hover:underline dark:text-blue-400">समाचार के प्रभाव का विश्लेषण करें →</button>
        `;
    } else {
        message = `
            <b>${news.headline}</b><br><br>
            ${news.summary}<br><br>
            <b>Market Impact:</b> ${news.impact}<br><br>
            <button onclick="analyzeNewsImpact('${newsId}')" class="text-sm text-blue-600 hover:underline dark:text-blue-400">Analyze News Impact →</button>
        `;
    }
    
    addAIMessage(message);
}

function analyzeNewsImpact(newsId) {
    let message = "";
    
    if (currentLanguage === 'hi') {
        message = `
            <b>समाचार प्रभाव विश्लेषण:</b><br><br>
            ${newsId === 'n1' ? 
            "RBI की नीति निर्णय बाजार की अपेक्षाओं के अनुरूप था। बैंकिंग शेयरों में मिश्रित प्रतिक्रिया देखी गई, जिसमें HDFC बैंक और ICICI बैंक में मामूली गिरावट देखी गई, जबकि SBI में मजबूती रही। रियल एस्टेट सेक्टर स्थिर रहा क्योंकि ब्याज दरों में कोई बदलाव नहीं हुआ है।" : 
            newsId === 'n2' ? 
            "इंफोसिस के बेहतर-से-अनुमानित परिणामों ने पूरे IT सेक्टर में सकारात्मक भावना को बढ़ावा दिया है। TCS और HCL टेक्नोलॉजीज जैसे अन्य IT शेयरों में भी 1-2% की बढ़त देखी गई। कंपनी का 4-7% की राजस्व वृद्धि मार्गदर्शन IT सेक्टर के लिए मध्यम सकारात्मक दृष्टिकोण का संकेत देता है।" :
            "वैश्विक बाजारों में तेजी से भारतीय बाजारों पर सकारात्मक प्रभाव पड़ने की उम्मीद है, विशेष रूप से IT शेयरों पर जो अमेरिकी बाजारों से मजबूत संबंध रखते हैं। निफ्टी IT इंडेक्स आज 1.5% ऊपर खुल सकता है। फेड रेट कट की उम्मीदों से FII निवेश में वृद्धि हो सकती है, जो भारतीय बाजारों के लिए सकारात्मक है।"}
            <br><br>
            <b>सुझाए गए कार्य:</b> ${newsId === 'n1' ? 
            "बैंकिंग शेयरों में किसी भी कमजोरी को खरीदारी के अवसर के रूप में देखें, विशेष रूप से HDFC बैंक और ICICI बैंक में।" : 
            newsId === 'n2' ? 
            "IT सेक्टर में लॉन्ग पोजीशन पर विचार करें, विशेष रूप से इंफोसिस और TCS में।" :
            "IT सेक्टर में मजबूत खरीदारी पर ध्यान दें, विशेष रूप से अमेरिकी एक्सपोजर वाले बड़े कैप स्टॉक्स में।"}
        `;
    } else {
        message = `
            <b>News Impact Analysis:</b><br><br>
            ${newsId === 'n1' ? 
            "RBI's policy decision was in line with market expectations. Banking stocks saw mixed reactions, with HDFC Bank and ICICI Bank seeing minor declines while SBI remained strong. Real estate sector stayed stable as there's no change in interest rates." : 
            newsId === 'n2' ? 
            "Infosys' better-than-expected results have boosted positive sentiment across the IT sector. Other IT stocks like TCS and HCL Technologies also saw 1-2% gains. The company's 4-7% revenue growth guidance signals moderately positive outlook for the IT sector." :
            "Global market rally is expected to have positive impact on Indian markets, particularly on IT stocks which have strong correlation with US markets. Nifty IT index may open 1.5% higher today. Fed rate cut hopes may increase FII inflows, which is positive for Indian markets."}
            <br><br>
            <b>Suggested Actions:</b> ${newsId === 'n1' ? 
            "Consider any weakness in banking stocks as buying opportunities, particularly in HDFC Bank and ICICI Bank." : 
            newsId === 'n2' ? 
            "Consider long positions in IT sector, particularly in Infosys and TCS." :
            "Focus on strong buying in IT sector, especially large cap stocks with US exposure."}
        `;
    }
    
    addAIMessage(message);
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    vantaEffect = initVantaBackground();
    
    userInput.addEventListener('keypress', function(e) {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });
    
    document.getElementById('darkModeToggle').addEventListener('click', function() {
        document.body.classList.toggle('dark');
        const icon = this.querySelector('i');
        if (document.body.classList.contains('dark')) {
            document.documentElement.style.backgroundColor = '#111827';
            vantaEffect.setOptions({ backgroundColor: 0x111827 });
        } else {
            document.documentElement.style.backgroundColor = '';
            vantaEffect.setOptions({ backgroundColor: 0xf1f5f9 });
        }
    });
});