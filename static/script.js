let currentAnalysisData = null; // A global variable to hold the analysis data

function logout() {
  localStorage.clear();
  window.location.href = "login.html";
}

async function analyzeJobOffer() {
  const jobOffer = document.getElementById('jobOffer').value;
  const resultsSection = document.getElementById('results');
  const riskScoreValue = document.getElementById('risk-score-value');
  const progressFill = document.getElementById('progress-fill');
  const progressBar = progressFill.parentElement;
  const riskLevel = document.getElementById('risk-level');
  const recommendation = document.getElementById('recommendation');
  const flagList = document.getElementById('flag-list');
  const checkBtn = document.querySelector('.check-btn');

  if (!jobOffer.trim()) {
    alert("Please paste a job offer to analyze.");
    document.getElementById('jobOffer').focus();
    return;
  }
  
  // Show loading state
  checkBtn.classList.add('loading');
  checkBtn.textContent = 'Analyzing...';
  checkBtn.disabled = true;

  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: jobOffer })
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();

    // Store the analysis data globally for the download function
    currentAnalysisData = data;
    window.currentAnalysisData = data;

    // Update UI with new data
    const scamRisk = Math.round(data.confidence * 100);
    const isScam = data.is_scam;
    
    resultsSection.style.display = 'block';
    resultsSection.scrollIntoView({ behavior: 'smooth' });

    // Animate the risk score
    animateValue(riskScoreValue, 0, scamRisk, 1500, '%');
    
    // Update progress bar
    setTimeout(() => {
      progressFill.style.width = scamRisk + "%";
      progressFill.style.background = isScam ? 
        "linear-gradient(90deg, #ff4d4d 0%, #ff6b6b 100%)" : 
        "linear-gradient(90deg, #00d26a 0%, #00ff7f 100%)";
      
      progressBar.setAttribute('aria-valuenow', scamRisk);
    }, 500);
    
    riskLevel.textContent = data.risk_level;
    riskLevel.style.color = isScam ? '#ff4d4d' : '#00d26a';
    recommendation.textContent = data.recommendation;
    
    // Populate flags and change dot color based on scam status
    flagList.innerHTML = '';
    if (data.fraud_indicators && data.fraud_indicators.length > 0) {
        data.fraud_indicators.forEach((flag, index) => {
            const listItem = document.createElement('li');
            listItem.className = 'flag-item';
            listItem.textContent = flag;
            
            // Change dot color based on scam status
            listItem.style.setProperty('--dot-color', isScam ? '#ff4d4d' : '#00d26a');
            
            // Animate flags appearance
            listItem.style.opacity = '0';
            listItem.style.transform = 'translateY(10px)';
            flagList.appendChild(listItem);
            
            setTimeout(() => {
              listItem.style.transition = 'all 0.3s ease';
              listItem.style.opacity = '1';
              listItem.style.transform = 'translateY(0)';
            }, 100 * index);
        });
    } else {
        const listItem = document.createElement('li');
        listItem.className = 'flag-item';
        listItem.textContent = "No specific flags detected.";
        listItem.style.setProperty('--dot-color', isScam ? '#ff4d4d' : '#00d26a');
        flagList.appendChild(listItem);
    }

  } catch (error) {
    console.error("Prediction failed:", error);
    alert("Analysis failed. Please check if the server is running and try again.");
  } finally {
      // Revert loading state
      checkBtn.classList.remove('loading');
      checkBtn.textContent = 'Check Now';
      checkBtn.disabled = false;
  }
}

function animateValue(element, start, end, duration, suffix = '') {
  let startTimestamp = null;
  const step = (timestamp) => {
    if (!startTimestamp) startTimestamp = timestamp;
    const progress = Math.min((timestamp - startTimestamp) / duration, 1);
    const current = Math.floor(progress * (end - start) + start);
    element.textContent = current + suffix;
    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };
  window.requestAnimationFrame(step);
}

// Function to generate and download the report
function downloadReport() {
    const data = currentAnalysisData;
    if (!data) {
      alert("Please analyze a job offer first.");
      return;
    }

    const reportContent = `
ScamScreener Analysis Report
============================

Date: ${new Date().toLocaleString()}

Original Text:
--------------
${document.getElementById('jobOffer').value}

Analysis Summary:
-----------------
Scam Risk Score: ${Math.round(data.confidence * 100)}%
Risk Level: ${data.risk_level}
Recommendation: ${data.recommendation}
${data.suggested_action ? 'Suggested Action: ' + data.suggested_action : ''}

Detailed Breakdown:
-------------------
Fraud Indicators Detected (${data.indicator_count || 0}):
${data.fraud_indicators && data.fraud_indicators.length > 0 ? 
  data.fraud_indicators.map(flag => `• ${flag}`).join('\n') : 
  '• None'}

Technical Details:
------------------
ML Score: ${data.ml_score || 'N/A'}
Pattern Score: ${data.pattern_score || 'N/A'}

---
Generated by ScamScreener - Protecting you from digital threats
`;

    const blob = new Blob([reportContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `scamscreener_report_${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
}

// Enhanced button click feedback
document.querySelectorAll('.action-btn').forEach(btn => {
  btn.addEventListener('click', function(e) {
    // Create ripple effect
    const ripple = document.createElement('span');
    const rect = this.getBoundingClientRect();
    const size = Math.max(rect.width, rect.height);
    const x = e.clientX - rect.left - size / 2;
    const y = e.clientY - rect.top - size / 2;
    
    ripple.style.cssText = `
      position: absolute;
      width: ${size}px;
      height: ${size}px;
      left: ${x}px;
      top: ${y}px;
      background: rgba(255,255,255,0.3);
      border-radius: 50%;
      transform: scale(0);
      animation: ripple 0.6s linear;
      pointer-events: none;
    `;
    
    this.style.position = 'relative';
    this.style.overflow = 'hidden';
    this.appendChild(ripple);
    
    setTimeout(() => ripple.remove(), 600);
  });
});

// Add ripple animation
const style = document.createElement('style');
style.textContent = `
  @keyframes ripple {
    to {
      transform: scale(4);
      opacity: 0;
    }
  }
`;
document.head.appendChild(style);

// Auto-resize textarea
const textarea = document.getElementById('jobOffer');
textarea.addEventListener('input', function() {
  this.style.height = 'auto';
  this.style.height = Math.max(150, this.scrollHeight) + 'px';
});
