// Function to handle user logout
function logout() {
  // Clear any stored authentication tokens or user data
  localStorage.removeItem('authToken');
  localStorage.removeItem('userData');
  
  // Redirect to homepage or login page
  window.location.href = 'index.html';
}

// Check authentication status and update navigation
function updateNavigation() {
  const authToken = localStorage.getItem('authToken');
  const loginLink = document.getElementById('nav-login-link');
  const signupLink = document.getElementById('nav-signup-link');
  const logoutLink = document.getElementById('nav-logout-link');
  
  if (authToken) {
    // User is logged in
    if (loginLink) loginLink.style.display = 'none';
    if (signupLink) signupLink.style.display = 'none';
    if (logoutLink) logoutLink.style.display = 'block';
  } else {
    // User is not logged in
    if (loginLink) loginLink.style.display = 'block';
    if (signupLink) signupLink.style.display = 'block';
    if (logoutLink) logoutLink.style.display = 'none';
  }
}

// Initialize navigation on page load
document.addEventListener('DOMContentLoaded', function() {
  updateNavigation();
  
  // Add logout functionality
  const logoutLink = document.getElementById('nav-logout-link');
  if (logoutLink) {
    logoutLink.addEventListener('click', function(e) {
      e.preventDefault();
      logout();
    });
  }
});
