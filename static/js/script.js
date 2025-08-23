function getScamTypeBadge(scamType) {
  if (!scamType) return '<span class="scam-type-badge scam-other">Unknown</span>';
  
  const type = scamType.toLowerCase();
  let className = 'scam-type-badge ';
  
  if (type.includes('phishing')) className += 'scam-phishing';
  else if (type.includes('investment')) className += 'scam-investment';
  else if (type.includes('romance')) className += 'scam-romance';
  else if (type.includes('tech')) className += 'scam-tech';
  else className += 'scam-other';
  
  return `<span class="${className}">${scamType}</span>`;
}

function createMessageCell(message, maxLength = 50) {
  if (!message) return '-';
  
  if (message.length <= maxLength) {
    return `<span class="message-text">${message}</span>`;
  }
  
  const truncated = message.substring(0, maxLength);
  const messageId = 'msg_' + Math.random().toString(36).substr(2, 9);
  
  return `
    <span class="message-text truncated" 
          id="${messageId}" 
          onclick="toggleMessage('${messageId}')" 
          title="Click to expand/collapse">
      ${truncated}...<span class="expand-indicator">▼</span>
    </span>
    <span class="full-message" style="display: none;">${message}</span>
  `;
}

function toggleMessage(messageId) {
  const element = document.getElementById(messageId);
  const fullMessage = element.nextElementSibling.textContent;
  const isExpanded = element.classList.contains('expanded');
  
  if (isExpanded) {
    // Collapse
    element.classList.remove('expanded');
    element.classList.add('truncated');
    element.innerHTML = fullMessage.substring(0, 50) + '...<span class="expand-indicator">▼</span>';
    element.title = "Click to expand/collapse";
  } else {
    // Expand
    element.classList.remove('truncated');
    element.classList.add('expanded');
    element.innerHTML = fullMessage + '<span class="expand-indicator">▲</span>';
    element.title = "Click to collapse";
  }
}

function truncateText(text, maxLength = 50) {
  if (!text) return '-';
  return text.length > maxLength ? text.substring(0, maxLength) + '...' : text;
}

function formatDate(dateString) {
  try {
    return new Date(dateString).toLocaleString();
  } catch (e) {
    return 'Invalid date';
  }
}

function animateCounter(elementId, targetValue) {
  const element = document.getElementById(elementId);
  const startValue = parseInt(element.textContent) || 0;
  const duration = 1000;
  const increment = (targetValue - startValue) / (duration / 16);
  let currentValue = startValue;
  
  const timer = setInterval(() => {
    currentValue += increment;
    if (currentValue >= targetValue) {
      element.textContent = targetValue;
      clearInterval(timer);
    } else {
      element.textContent = Math.floor(currentValue);
    }
  }, 16);
}

async function fetchDashboardData() {
  const token = localStorage.getItem("access_token");
  if (!token) {
    alert("⚠️ Please log in first.");
    window.location.href = "login.html";
    return;
  }

  try {
    // Fetch counts
    const countsRes = await fetch("http://localhost:5000/dashboard/stats", {
      headers: { "Authorization": `Bearer ${token}` }
    });
    const counts = await countsRes.json();
    if (countsRes.ok) {
      animateCounter("totalUsers", counts.total_users);
      animateCounter("totalReports", counts.total_reports);
    }

    // Fetch reports
    const reportsRes = await fetch("http://localhost:5000/dashboard/reports", {
      headers: { "Authorization": `Bearer ${token}` }
    });
    const reports = await reportsRes.json();
    const reportsTable = document.getElementById("reportsTable");
    reportsTable.innerHTML = "";

    if (reportsRes.ok && reports.length > 0) {
      reports.forEach(r => {
        const tr = document.createElement("tr");
        tr.innerHTML = `
          <td><span class="table-id">#${r.id}</span></td>
          <td class="message-cell">${createMessageCell(r.message)}</td>
          <td>${getScamTypeBadge(r.scam_type)}</td>
          <td>${r.company || '-'}</td>
          <td>${formatDate(r.created_at)}</td>
        `;
        reportsTable.appendChild(tr);
      });
    } else {
      reportsTable.innerHTML = `<tr><td colspan="5" class="empty-state">No reports yet</td></tr>`;
    }
  } catch (err) {
    console.error("Dashboard fetch error:", err);
    const reportsTable = document.getElementById("reportsTable");
    reportsTable.innerHTML = `<tr><td colspan="5" class="empty-state">Failed to load reports</td></tr>`;
  }
}

function logout() {
  localStorage.clear();
  window.location.href = "login.html";
}

// Add event listener to logout link
document.addEventListener('DOMContentLoaded', function() {
  const logoutLink = document.getElementById('nav-logout-link');
  if (logoutLink) {
    logoutLink.addEventListener('click', logout);
  }
  
  // Fetch dashboard data when page loads
  fetchDashboardData();
});
