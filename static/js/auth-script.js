document.addEventListener('DOMContentLoaded', function() {
  const loginForm = document.getElementById('loginForm');
  
  if (loginForm) {
    loginForm.addEventListener('submit', async function(e) {
      e.preventDefault();
      
      const usernameOrEmail = document.getElementById('loginUsernameOrEmail').value.trim();
      const password = document.getElementById('loginPassword').value.trim();
      const errorMessage = document.getElementById('errorMessage');

      if (!usernameOrEmail || !password) {
        errorMessage.textContent = 'Please fill in all fields';
        errorMessage.style.display = 'block';
        return;
      }

      try {
        const response = await fetch('http://localhost:5000/auth/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            username_or_email: usernameOrEmail,
            password: password
          })
        });

        const data = await response.json();

        if (response.ok) {
          // Save tokens for later API calls
          localStorage.setItem('access_token', data.access_token);
          localStorage.setItem('refresh_token', data.refresh_token);
          localStorage.setItem('user', JSON.stringify(data.user));

          alert('✅ Login successful! Redirecting to dashboard...');
          window.location.href = 'dashboard.html';
        } else {
          errorMessage.textContent = data.error || 'Login failed';
          errorMessage.style.display = 'block';
        }
      } catch (error) {
        errorMessage.textContent = `An error occurred: ${error.message}`;
        errorMessage.style.display = 'block';
      }
    });
  }
});
